"""Secondary replay sampler used to mix selected simulator trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.reinforcement_learning.rwm_trace.v10_protocol import COMMAND_MODES, load_protocol
from scripts.reinforcement_learning.rwm_trace.v10_replay_manifest import verify_manifest


REPLAY_KEYS = ("observation", "action", "reward", "terminated", "truncated", "next_observation")


class TraceReplaySampler:
    def __init__(self, path: str | Path, *, seed: int = 0) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("format_version") not in {"go2_trace_replay_v1", "go2_trace_replay_v2"}:
            raise ValueError(f"Unsupported TRACE replay artifact: {path}")
        self.format_version = str(artifact.get("format_version"))
        self.data: dict[str, torch.Tensor] = {}
        for key in REPLAY_KEYS:
            value = artifact.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"TRACE replay is missing tensor {key!r}.")
            self.data[key] = value.detach().cpu()
        sizes = {int(value.shape[0]) for value in self.data.values()}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError("TRACE replay tensors must have one shared non-zero leading dimension.")
        self.num_transitions = next(iter(sizes))
        self.metadata = dict(artifact.get("metadata") or {})
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def sample(self, batch_size: int, *, device: torch.device | str) -> dict[str, torch.Tensor]:
        indices = torch.randint(self.num_transitions, (int(batch_size),), generator=self.generator)
        return {key: value[indices].to(device) for key, value in self.data.items()}


class V10TraceReplaySampler:
    """Sample retained V10 shards using their immutable sampling strategy."""

    def __init__(self, manifest_path: str | Path, *, seed: int = 0) -> None:
        manifest_path = Path(manifest_path).expanduser().resolve()
        manifest = verify_manifest(manifest_path, allow_pending=True)
        protocol, _, protocol_sha = load_protocol(manifest["protocol_path"])
        if manifest.get("protocol_sha256") != protocol_sha:
            raise ValueError("V10 replay manifest and protocol SHA differ.")
        self.manifest_path = manifest_path
        self.metadata = dict(manifest)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.protocol = protocol
        self.shards: list[dict[str, torch.Tensor]] = []
        bucket_targets: np.ndarray | None = None
        sampling_strategies: set[str] = set()
        for row in manifest["shards"]:
            payload = torch.load(row["path"], map_location="cpu", weights_only=False, mmap=True)
            if payload.get("format_version") != "go2_trace_replay_v10_shard_v1":
                raise ValueError(f"Manifest includes a non-V10 shard: {row['path']}")
            required = (*REPLAY_KEYS, "command_mode_id", "command_signed_buckets")
            missing = [key for key in required if not isinstance(payload.get(key), torch.Tensor)]
            if missing:
                raise ValueError(f"V10 replay shard is missing fields {missing}: {row['path']}")
            sizes = {int(payload[key].shape[0]) for key in required}
            if len(sizes) != 1:
                raise ValueError(f"V10 replay shard fields have inconsistent lengths: {row['path']}")
            self.shards.append({key: payload[key] for key in required})
            sampling_strategies.add(
                str(
                    (payload.get("metadata") or {}).get(
                        "sampling_strategy", "dataset_mode_signed_magnitude_stratified"
                    )
                )
            )
            current_targets = np.asarray(
                (payload.get("metadata") or {}).get("target_mode_bucket_probabilities"),
                dtype=np.float64,
            )
            if current_targets.shape != (len(COMMAND_MODES), 3, 6):
                raise ValueError(f"V10 shard has invalid mode-bucket targets: {row['path']}")
            if bucket_targets is None:
                bucket_targets = current_targets
            elif not np.allclose(bucket_targets, current_targets, rtol=0.0, atol=1.0e-12):
                raise ValueError("Retained V10 shards use different command magnitude targets.")
        if not self.shards or bucket_targets is None:
            raise ValueError("V10 replay manifest contains no shards.")
        if len(sampling_strategies) != 1:
            raise ValueError("Retained V10 shards use different sampling strategies.")
        self.sampling_strategy = sampling_strategies.pop()
        if self.sampling_strategy not in {
            "dataset_mode_signed_magnitude_stratified",
            "uniform_selected_replay",
        }:
            raise ValueError(f"Unsupported V10 sampling strategy: {self.sampling_strategy}")
        if manifest.get("sampling_strategy", self.sampling_strategy) != self.sampling_strategy:
            raise ValueError("V10 manifest and shard sampling strategies differ.")
        self.num_transitions = sum(int(shard["reward"].shape[0]) for shard in self.shards)
        self.observation_dim = int(self.shards[0]["observation"].shape[-1])
        self.action_dim = int(self.shards[0]["action"].shape[-1])
        self.n_step = int(protocol["replay"]["n_step"])
        self.gamma = float(protocol["replay"]["gamma"])
        self.uniform_catalog: dict[str, torch.Tensor] | None = None
        self.catalogs: list[dict[str, torch.Tensor]] = []
        if self.sampling_strategy == "uniform_selected_replay":
            shard_ids = []
            row_ids = []
            for shard_id, shard in enumerate(self.shards):
                count = int(shard["reward"].shape[0])
                shard_ids.append(torch.full((count,), shard_id, dtype=torch.long))
                row_ids.append(torch.arange(count, dtype=torch.long))
            self.uniform_catalog = {
                "shard_ids": torch.cat(shard_ids),
                "row_ids": torch.cat(row_ids),
            }
            self.mode_probabilities = np.empty(0, dtype=np.float64)
            self.mode_carry = np.empty(0, dtype=np.float64)
        else:
            mode_counts = protocol["candidate"]["selected_mode_counts"]
            self.mode_probabilities = np.asarray(
                [float(mode_counts[mode]) for mode in COMMAND_MODES], dtype=np.float64
            )
            self.mode_probabilities /= self.mode_probabilities.sum()
            self.mode_carry = np.zeros(len(COMMAND_MODES), dtype=np.float64)
            self.catalogs = self._build_catalogs(bucket_targets)

    def _build_catalogs(self, targets: np.ndarray) -> list[dict[str, torch.Tensor]]:
        catalogs: list[dict[str, torch.Tensor]] = []
        for mode_id, mode in enumerate(COMMAND_MODES):
            shard_ids: list[torch.Tensor] = []
            row_ids: list[torch.Tensor] = []
            bucket_rows: list[torch.Tensor] = []
            for shard_id, shard in enumerate(self.shards):
                rows = torch.nonzero(shard["command_mode_id"].reshape(-1) == mode_id, as_tuple=False).flatten()
                if len(rows):
                    shard_ids.append(torch.full((len(rows),), shard_id, dtype=torch.long))
                    row_ids.append(rows.long())
                    bucket_rows.append(shard["command_signed_buckets"][rows].to(torch.int16))
            if not row_ids:
                raise ValueError(f"Retained V10 replay has no transitions for command mode {mode}.")
            shard_id_t = torch.cat(shard_ids)
            row_id_t = torch.cat(row_ids)
            buckets = torch.cat(bucket_rows).numpy()
            weights = np.ones(len(row_id_t), dtype=np.float64)
            active_axes = [axis for axis in range(3) if targets[mode_id, axis].sum() > 0.0]
            for _ in range(100):
                maximum_error = 0.0
                for axis in active_axes:
                    target = targets[mode_id, axis]
                    axis_total = float(weights.sum())
                    for bucket in range(6):
                        mask = buckets[:, axis] == bucket
                        current = float(weights[mask].sum())
                        desired = float(target[bucket] * axis_total)
                        if target[bucket] > 0.0 and current <= 0.0:
                            raise ValueError(
                                f"Mode {mode} has no replay support for axis={axis}, signed_bucket={bucket}."
                            )
                        if current > 0.0:
                            weights[mask] *= desired / current
                    normalized = np.asarray(
                        [weights[buckets[:, axis] == bucket].sum() for bucket in range(6)]
                    )
                    normalized /= max(normalized.sum(), 1.0e-12)
                    maximum_error = max(maximum_error, float(np.max(np.abs(normalized - target))))
                if maximum_error < 1.0e-6:
                    break
            if not np.isfinite(weights).all() or weights.sum() <= 0.0:
                raise ValueError(f"Could not calibrate V10 sampling weights for command mode {mode}.")
            weights /= weights.sum()
            catalogs.append(
                {
                    "shard_ids": shard_id_t,
                    "row_ids": row_id_t,
                    "weights": torch.as_tensor(weights, dtype=torch.float64),
                }
            )
        return catalogs

    def _mode_counts(self, batch_size: int) -> np.ndarray:
        desired = self.mode_probabilities * int(batch_size) + self.mode_carry
        counts = np.floor(desired).astype(np.int64)
        remainder = int(batch_size - counts.sum())
        order = sorted(range(len(counts)), key=lambda index: (-(desired[index] - counts[index]), index))
        for index in order[:remainder]:
            counts[index] += 1
        self.mode_carry = desired - counts
        if int(counts.sum()) != int(batch_size) or np.any(counts < 0):
            raise RuntimeError(f"Invalid V10 per-mode batch allocation: {counts.tolist()}")
        return counts

    def sample(self, batch_size: int, *, device: torch.device | str) -> dict[str, torch.Tensor]:
        if self.sampling_strategy == "uniform_selected_replay":
            if self.uniform_catalog is None:
                raise RuntimeError("Uniform V10 replay catalog was not initialized.")
            positions = torch.randint(
                self.num_transitions,
                (int(batch_size),),
                generator=self.generator,
            )
            selected_shards = self.uniform_catalog["shard_ids"][positions]
            selected_rows = self.uniform_catalog["row_ids"][positions]
            chunks: dict[str, list[torch.Tensor]] = {key: [] for key in REPLAY_KEYS}
            for shard_id in torch.unique(selected_shards).tolist():
                rows = selected_rows[selected_shards == int(shard_id)]
                shard = self.shards[int(shard_id)]
                for key in REPLAY_KEYS:
                    chunks[key].append(shard[key][rows])
            result = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
            permutation = torch.randperm(int(batch_size), generator=self.generator)
            return {key: value[permutation].to(device) for key, value in result.items()}
        chunks: dict[str, list[torch.Tensor]] = {key: [] for key in REPLAY_KEYS}
        for mode_id, count in enumerate(self._mode_counts(int(batch_size))):
            if count == 0:
                continue
            catalog = self.catalogs[mode_id]
            positions = torch.multinomial(
                catalog["weights"], int(count), replacement=True, generator=self.generator
            )
            selected_shards = catalog["shard_ids"][positions]
            selected_rows = catalog["row_ids"][positions]
            for shard_id in torch.unique(selected_shards).tolist():
                mask = selected_shards == int(shard_id)
                rows = selected_rows[mask]
                shard = self.shards[int(shard_id)]
                for key in REPLAY_KEYS:
                    chunks[key].append(shard[key][rows])
        result = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
        if int(result["reward"].shape[0]) != int(batch_size):
            raise RuntimeError("V10 replay sampler returned the wrong batch size.")
        permutation = torch.randperm(int(batch_size), generator=self.generator)
        return {key: value[permutation].to(device) for key, value in result.items()}


def mix_trace_replay_batch(
    batch: dict[str, torch.Tensor],
    sampler: TraceReplaySampler | None,
    ratio: float,
) -> tuple[dict[str, torch.Tensor], int]:
    """Replace a fixed fraction of a sampled RWM batch with TRACE replay."""

    if sampler is None or ratio <= 0.0:
        return batch, 0
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError(f"TRACE replay ratio must be in [0, 1], got {ratio}.")
    batch_size = int(batch["reward"].shape[0])
    trace_count = min(batch_size, int(round(batch_size * float(ratio))))
    if trace_count == 0:
        return batch, 0
    trace_batch = sampler.sample(trace_count, device=batch["reward"].device)
    mixed = dict(batch)
    for key in REPLAY_KEYS:
        if key not in mixed:
            raise KeyError(f"Primary replay batch is missing {key!r}.")
        if tuple(mixed[key].shape[1:]) != tuple(trace_batch[key].shape[1:]):
            raise ValueError(
                f"TRACE replay shape mismatch for {key}: {tuple(trace_batch[key].shape)} vs {tuple(mixed[key].shape)}"
            )
        mixed[key] = mixed[key].clone()
        mixed[key][:trace_count] = trace_batch[key].to(dtype=mixed[key].dtype)
    return mixed, trace_count
