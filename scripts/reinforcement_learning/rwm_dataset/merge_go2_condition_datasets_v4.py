#!/usr/bin/env python3
"""Merge condition-specific Go2 datasets without crossing episode boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


MODE_NAMES = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")
MODE_IDS = {
    (False, False, False): 0,
    (True, False, False): 1,
    (False, True, False): 2,
    (False, False, True): 3,
    (True, True, False): 4,
    (True, False, True): 5,
    (False, True, True): 6,
    (True, True, True): 7,
}
REQUIRED_KEYS = {
    "states", "actions", "next_states", "contacts", "terminations",
    "commands", "episode_ids", "timesteps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--input", action="append", required=True, metavar="NAME=PATH",
        help="Condition name and selected dataset path; repeat in desired condition-id order.",
    )
    parser.add_argument(
        "--expected-count", action="append", required=True, metavar="NAME=COUNT",
        help="Exact transition count expected for each condition.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--zero-epsilon", type=float, default=1.0e-3)
    return parser.parse_args()


def parse_mapping(values: list[str], cast: type = str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        name, raw = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate name in {value!r}")
        result[name] = cast(raw)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stack(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list) and value:
        return torch.stack(value)
    raise TypeError(f"Expected a non-empty tensor/list transition field, got {type(value).__name__}")


def transition_count(dataset: dict[str, Any]) -> int:
    return int(stack(dataset["states"]).shape[0])


def is_transition_field(value: Any, count: int) -> bool:
    return (
        isinstance(value, list) and len(value) == count
    ) or (
        isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == count
    )


def item(value: Any, index: int) -> torch.Tensor:
    tensor = value[index]
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    # Canonical selected datasets use one environment. Preserve that leading axis.
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.detach().cpu().clone()


def scalar_series(dataset: dict[str, Any], key: str) -> np.ndarray:
    return stack(dataset[key]).detach().cpu().reshape(transition_count(dataset), -1)[:, 0].numpy()


def mode_counts(commands: torch.Tensor, epsilon: float) -> dict[str, int]:
    values = commands.detach().cpu().reshape(-1, 3).numpy()
    active = np.abs(values) > float(epsilon)
    ids = np.full(len(values), -1, dtype=np.int8)
    for bits, mode_id in MODE_IDS.items():
        ids[np.all(active == np.asarray(bits), axis=-1)] = mode_id
    if (ids < 0).any():
        raise ValueError(f"Found {int((ids < 0).sum())} commands outside the eight-mode taxonomy")
    return {name: int((ids == index).sum()) for index, name in enumerate(MODE_NAMES)}


def base_lin_vel_is_supervised(metadata: dict[str, Any], state_dim: int) -> bool:
    if "base_lin_vel_supervised" in metadata:
        return bool(metadata["base_lin_vel_supervised"])
    # Native MJLab collection stores a 48-D full observation
    # [state45, command3].  Its 45-D state therefore includes measured
    # simulator base linear velocity even in older files that predate the
    # explicit provenance flag.
    return state_dim == 45 and int(metadata.get("full_rwm_obs_dim", -1)) == 48


def main() -> None:
    args = parse_args()
    input_map = parse_mapping(args.input, Path)
    expected = parse_mapping(args.expected_count, int)
    if set(input_map) != set(expected):
        raise ValueError(f"Input and expected-count names differ: {set(input_map) ^ set(expected)}")

    loaded: list[tuple[str, Path, dict[str, Any], int]] = []
    common_transition_keys: set[str] | None = None
    state_dim = action_dim = None
    for name, raw_path in input_map.items():
        path = raw_path.expanduser().resolve()
        dataset = torch.load(path, map_location="cpu", weights_only=False)
        missing = REQUIRED_KEYS - set(dataset)
        if missing:
            raise KeyError(f"{name} is missing required fields: {sorted(missing)}")
        count = transition_count(dataset)
        if count != expected[name]:
            raise ValueError(f"{name} has {count} transitions, expected {expected[name]}")
        keys = {key for key, value in dataset.items() if is_transition_field(value, count)}
        common_transition_keys = keys if common_transition_keys is None else common_transition_keys & keys
        current_state_dim = int(stack(dataset["states"]).shape[-1])
        current_action_dim = int(stack(dataset["actions"]).shape[-1])
        state_dim = current_state_dim if state_dim is None else state_dim
        action_dim = current_action_dim if action_dim is None else action_dim
        if current_state_dim != state_dim or current_action_dim != action_dim:
            raise ValueError(
                f"Dimension mismatch for {name}: state={current_state_dim}, action={current_action_dim}; "
                f"expected state={state_dim}, action={action_dim}"
            )
        loaded.append((name, path, dataset, count))

    assert common_transition_keys is not None
    if not REQUIRED_KEYS <= common_transition_keys:
        raise ValueError(f"Required fields are not transition-aligned in every input: {sorted(REQUIRED_KEYS - common_transition_keys)}")
    common_transition_keys -= {"episode_ids", "timesteps", "condition_ids"}
    output: dict[str, Any] = {key: [] for key in sorted(common_transition_keys)}
    output.update({"episode_ids": [], "timesteps": [], "condition_ids": []})
    next_episode = 0
    condition_reports: dict[str, Any] = {}

    for condition_id, (name, path, dataset, count) in enumerate(loaded):
        episodes = scalar_series(dataset, "episode_ids")
        timesteps = scalar_series(dataset, "timesteps")
        local_runs = 0
        local_timestep = 0
        previous_episode = previous_timestep = None
        for index in range(count):
            contiguous = (
                index > 0
                and episodes[index] == previous_episode
                and timesteps[index] == previous_timestep + 1
            )
            if not contiguous:
                if index > 0:
                    next_episode += 1
                local_runs += 1
                local_timestep = 0
            for key in common_transition_keys:
                output[key].append(item(dataset[key], index))
            output["episode_ids"].append(torch.tensor([next_episode], dtype=torch.long))
            output["timesteps"].append(torch.tensor([local_timestep], dtype=torch.long))
            output["condition_ids"].append(torch.tensor([condition_id], dtype=torch.long))
            local_timestep += 1
            previous_episode = episodes[index]
            previous_timestep = timesteps[index]
        next_episode += 1
        commands = stack(dataset["commands"])
        condition_reports[name] = {
            "condition_id": condition_id,
            "input": str(path),
            "input_sha256": sha256(path),
            "transitions": count,
            "fraction": count / sum(expected.values()),
            "contiguous_runs": local_runs,
            "command_mode_counts": mode_counts(commands, args.zero_epsilon),
            "source_metadata": dict(dataset.get("metadata") or {}),
        }

    total = sum(expected.values())
    if len(output["states"]) != total:
        raise RuntimeError(f"Merged {len(output['states'])} transitions, expected {total}")
    obs_dim = int(stack(output["observations"]).shape[-1]) if "observations" in output else state_dim
    contact_dim = int(stack(output["contacts"]).shape[-1])
    termination_dim = int(stack(output["terminations"]).shape[-1])
    source_metadata = [dict(dataset.get("metadata") or {}) for _, _, dataset, _ in loaded]
    dt_values = {float(metadata["dt"]) for metadata in source_metadata if "dt" in metadata}
    if len(dt_values) > 1:
        raise ValueError(f"Source dt values differ: {sorted(dt_values)}")
    output["num_envs"] = 1
    output["capacity"] = total
    output["metadata"] = {
        "schema": "go2_pooled_gap_dataset_v4",
        "selection_kind": "weighted_condition_contiguous_segments_v4",
        "condition_id": "pooled_40_20_20_10_10",
        "condition_id_map": {name: report["condition_id"] for name, report in condition_reports.items()},
        "condition_counts": {name: report["transitions"] for name, report in condition_reports.items()},
        "condition_fractions": {name: report["fraction"] for name, report in condition_reports.items()},
        "num_transitions": total,
        "num_episodes": next_episode,
        "num_time_steps": total,
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "contact_dim": contact_dim,
        "termination_dim": termination_dim,
        "dt": next(iter(dt_values)) if dt_values else 0.02,
        "dataset_obs_kind": source_metadata[0].get("dataset_obs_kind", "proprioceptive"),
        "full_rwm_obs_dim": source_metadata[0].get("full_rwm_obs_dim", state_dim + 3),
        "live_actor_obs_dim": source_metadata[0].get("live_actor_obs_dim", obs_dim),
        "base_lin_vel_supervised": all(
            base_lin_vel_is_supervised(dict(dataset.get("metadata") or {}), int(state_dim))
            for _, _, dataset, _ in loaded
        ),
        "source_sha256": {name: report["input_sha256"] for name, report in condition_reports.items()},
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    merged_modes = mode_counts(stack(output["commands"]), args.zero_epsilon)
    report = {
        "schema": "go2_pooled_gap_merge_v4",
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "total_transitions": total,
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "contact_dim": contact_dim,
        "termination_dim": termination_dim,
        "num_episodes": next_episode,
        "base_lin_vel_supervised": output["metadata"]["base_lin_vel_supervised"],
        "condition_reports": condition_reports,
        "merged_command_mode_counts": merged_modes,
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output_path),
        "total_transitions": total,
        "condition_counts": output["metadata"]["condition_counts"],
        "command_mode_counts": merged_modes,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "base_lin_vel_supervised": output["metadata"]["base_lin_vel_supervised"],
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
