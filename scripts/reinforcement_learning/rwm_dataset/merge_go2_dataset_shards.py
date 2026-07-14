"""Merge independently-created Go2 dataset shards without crossing trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


_COMPATIBILITY_METADATA_KEYS = (
    "task",
    "robot",
    "obs_dim",
    "dataset_obs_kind",
    "full_rwm_obs_dim",
    "action_dim",
    "state_dim",
    "contact_dim",
    "termination_dim",
    "live_actor_obs_dim",
    "live_critic_obs_dim",
    "actuator_names",
    "joint_names",
    "collector_mix",
    "expert_policy_path",
    "medium_policy_path",
    "action_noise_std",
    "medium_action_noise_std",
    "failure_action_noise_std",
    "command_modes",
    "command_mode_weights",
    "command_ranges",
    "command_resample_interval",
    "broken_pd_joint_names",
    "joint_strength_scales",
    "env_randomization",
    "env_step_action_interface",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_shards", type=int, default=4)
    parser.add_argument("--minimum_transitions", type=int, default=1_000_000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _validate_compatible(reference: dict[str, Any], candidate: dict[str, Any], path: Path) -> None:
    for key in ("state_dim", "action_dim", "contact_dim", "termination_dim", "num_envs"):
        if candidate.get(key) != reference.get(key):
            raise ValueError(f"Shard {path} has incompatible {key}: {candidate.get(key)!r}")
    reference_meta = reference.get("metadata") or {}
    candidate_meta = candidate.get("metadata") or {}
    for key in _COMPATIBILITY_METADATA_KEYS:
        if _canonical_json(candidate_meta.get(key)) != _canonical_json(reference_meta.get(key)):
            raise ValueError(f"Shard {path} has incompatible metadata[{key!r}]")


def _validate_time_lists(dataset: dict[str, Any], path: Path) -> list[str]:
    list_keys = [key for key, value in dataset.items() if isinstance(value, list)]
    if "states" not in list_keys or "episode_ids" not in list_keys:
        raise ValueError(f"Shard {path} is missing states or episode_ids")
    steps = len(dataset["states"])
    num_envs = int(dataset["num_envs"])
    for key in list_keys:
        values = dataset[key]
        if len(values) != steps:
            raise ValueError(f"Shard {path} key {key!r} has {len(values)} rows, expected {steps}")
        for row in values:
            if not isinstance(row, torch.Tensor) or int(row.shape[0]) != num_envs:
                raise ValueError(f"Shard {path} key {key!r} contains a malformed vector row")
    return list_keys


def _offset_time_rows(rows: list[torch.Tensor], offset: int) -> list[torch.Tensor]:
    return [row.long() + int(offset) for row in rows]


def _merge_named_tables(
    tables: list[dict[str, Any]],
    *,
    id_key: str,
) -> dict[str, Any]:
    reference_keys = set(tables[0])
    for table in tables[1:]:
        if set(table) != reference_keys:
            raise ValueError(f"Table {id_key!r} schemas differ across shards")
    merged: dict[str, Any] = {}
    for key in sorted(reference_keys):
        values = [table[key] for table in tables]
        if all(isinstance(value, torch.Tensor) for value in values):
            merged[key] = torch.cat(values, dim=0)
        else:
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"Table metadata {key!r} differs across shards")
            merged[key] = copy.deepcopy(values[0])
    ids = merged[id_key]
    if not isinstance(ids, torch.Tensor) or int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError(f"Merged table {id_key!r} is not globally unique")
    return merged


def merge_shards(paths: list[Path]) -> dict[str, Any]:
    datasets = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    reference = datasets[0]
    reference_list_keys = set(_validate_time_lists(reference, paths[0]))
    for path, dataset in zip(paths[1:], datasets[1:]):
        _validate_compatible(reference, dataset, path)
        if set(_validate_time_lists(dataset, path)) != reference_list_keys:
            raise ValueError(f"Shard {path} has a different transition schema")

    prepared: list[dict[str, Any]] = []
    startup_tables: list[dict[str, Any]] = []
    episode_tables: list[dict[str, Any]] = []
    episode_offset = 0
    startup_offset = 0
    shard_boundaries = [0]
    for path, source in zip(paths, datasets):
        dataset = copy.deepcopy(source)
        startup_table = copy.deepcopy(dataset.pop("startup_domain_table"))
        episode_table = copy.deepcopy(dataset.pop("episode_domain_table"))

        dataset["episode_ids"] = _offset_time_rows(dataset["episode_ids"], episode_offset)
        episode_table["episode_id"] = episode_table["episode_id"].long() + episode_offset
        startup_table["startup_domain_id"] = startup_table["startup_domain_id"].long() + startup_offset
        episode_table["startup_domain_id"] = episode_table["startup_domain_id"].long() + startup_offset

        local_episode_max = max(
            max(int(row.max().item()) for row in source["episode_ids"]),
            int(source["episode_domain_table"]["episode_id"].max().item()),
        )
        local_startup_rows = int(startup_table["startup_domain_id"].numel())
        episode_offset += local_episode_max + 1
        startup_offset += local_startup_rows
        shard_boundaries.append(shard_boundaries[-1] + len(dataset["states"]))
        prepared.append(dataset)
        startup_tables.append(startup_table)
        episode_tables.append(episode_table)

    merged = copy.deepcopy(prepared[0])
    for key in reference_list_keys:
        merged[key] = []
        for dataset in prepared:
            merged[key].extend(dataset[key])
    merged["startup_domain_table"] = _merge_named_tables(
        startup_tables,
        id_key="startup_domain_id",
    )
    merged["episode_domain_table"] = _merge_named_tables(
        episode_tables,
        id_key="episode_id",
    )

    num_steps = len(merged["states"])
    num_transitions = num_steps * int(merged["num_envs"])
    metadata = copy.deepcopy(reference.get("metadata") or {})
    source_metadata = [dataset.get("metadata") or {} for dataset in datasets]
    collector_names = set().union(*(meta.get("collector_counts", {}).keys() for meta in source_metadata))
    mode_names = set().union(*(meta.get("command_mode_counts", {}).keys() for meta in source_metadata))
    completed_episodes = sum(int(meta.get("completed_episode_count", 0)) for meta in source_metadata)

    def weighted_metric(key: str) -> float:
        if completed_episodes > 0:
            return sum(
                float(meta.get(key, 0.0)) * int(meta.get("completed_episode_count", 0))
                for meta in source_metadata
            ) / completed_episodes
        return sum(float(meta.get(key, 0.0)) for meta in source_metadata) / len(source_metadata)

    metadata.update(
        {
            "merged_shards": len(paths),
            "num_time_steps": num_steps,
            "num_transitions": num_transitions,
            "actual_num_transitions": num_transitions,
            "shard_boundaries_time_steps": shard_boundaries,
            "collector_counts": {
                name: sum(int(meta.get("collector_counts", {}).get(name, 0)) for meta in source_metadata)
                for name in sorted(collector_names)
            },
            "command_mode_counts": {
                name: sum(int(meta.get("command_mode_counts", {}).get(name, 0)) for meta in source_metadata)
                for name in sorted(mode_names)
            },
            "mean_reward": weighted_metric("mean_reward"),
            "mean_episode_length": weighted_metric("mean_episode_length"),
            "completed_episode_count": completed_episodes,
            "termination_count": sum(int(meta.get("termination_count", 0)) for meta in source_metadata),
            "timeout_count": sum(int(meta.get("timeout_count", 0)) for meta in source_metadata),
            "collection_seconds": sum(float(meta.get("collection_seconds", 0.0)) for meta in source_metadata),
            "source_shards": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "seed": int((dataset.get("metadata") or {}).get("seed", -1)),
                    "shard_id": int((dataset.get("metadata") or {}).get("shard_id", index)),
                    "num_time_steps": len(dataset["states"]),
                    "num_transitions": len(dataset["states"]) * int(dataset["num_envs"]),
                }
                for index, (path, dataset) in enumerate(zip(paths, datasets))
            ],
        }
    )
    merged["metadata"] = metadata
    merged["capacity"] = num_transitions
    return merged


def main() -> None:
    args = _parse_args()
    paths = [Path(value).expanduser().resolve() for value in args.inputs]
    if len(paths) != int(args.expected_shards):
        raise ValueError(f"Expected {args.expected_shards} shards, got {len(paths)}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shards: {missing}")

    merged = merge_shards(paths)
    actual = int(merged["metadata"]["actual_num_transitions"])
    if actual < int(args.minimum_transitions):
        raise ValueError(f"Merged dataset has {actual} transitions, below {args.minimum_transitions}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"merged_dataset={output}")
    print(f"sha256={_sha256(output)}")
    print(f"num_shards={len(paths)}")
    print(f"num_time_steps={len(merged['states'])}")
    print(f"num_transitions={actual}")
    print(f"startup_domains={merged['startup_domain_table']['startup_domain_id'].numel()}")
    print(f"episodes={merged['episode_domain_table']['episode_id'].numel()}")


if __name__ == "__main__":
    main()
