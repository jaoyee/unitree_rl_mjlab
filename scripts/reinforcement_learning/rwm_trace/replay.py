"""Secondary replay sampler used to mix selected simulator trajectories."""

from __future__ import annotations

from pathlib import Path

import torch


REPLAY_KEYS = ("observation", "action", "reward", "terminated", "truncated", "next_observation")


class TraceReplaySampler:
    def __init__(self, path: str | Path, *, seed: int = 0) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("format_version") != "go2_trace_replay_v1":
            raise ValueError(f"Unsupported TRACE replay artifact: {path}")
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
