"""Strict validator for final matched real/sim Go2 RWM datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--block-spec", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-transitions", type=int, default=25_000)
    parser.add_argument("--require-expert", action="store_true")
    parser.add_argument("--minimum-valid-40-step-starts", type=int, default=9000)
    return parser.parse_args()


def stack(value: Any, key: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.stack(value)
    if not torch.isfinite(tensor.float()).all():
        raise ValueError(f"{key} contains NaN/Inf")
    return tensor


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser()
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    spec = json.loads(Path(args.block_spec).expanduser().read_text()) if args.block_spec else None
    expected_shapes = {
        "states": (1, 45), "actions": (1, 12), "next_states": (1, 45),
        "contacts": (1, 4), "terminations": (1, 1),
        "observations": (1, 45), "next_observations": (1, 45),
    }
    n = len(data["states"])
    if n != args.expected_transitions:
        raise ValueError(f"dataset has {n} transitions, expected {args.expected_transitions}")
    tensors = {}
    for key, shape in expected_shapes.items():
        tensors[key] = stack(data[key], key)
        if tuple(tensors[key].shape[1:]) != shape:
            raise ValueError(f"{key} shape {tuple(tensors[key].shape)}, expected [N,{shape}]")
    for key, value in data.items():
        if isinstance(value, list) and len(value) == n:
            stack(value, key)
    actions = tensors["actions"].reshape(n, 12)
    if float(actions.abs().max()) > 1.0001:
        raise ValueError("normalized action exceeds [-1,1]")
    episode = stack(data["episode_ids"], "episode_ids").reshape(n)
    timestep = stack(data["timesteps"], "timesteps").reshape(n)
    run_lengths = []
    start = 0
    for index in range(1, n + 1):
        contiguous = (
            index < n
            and int(episode[index]) == int(episode[index - 1])
            and int(timestep[index]) == int(timestep[index - 1]) + 1
        )
        if contiguous:
            continue
        run_lengths.append(index - start)
        start = index
    minimum_block = int(spec["minimum_block"]) if spec else None
    if minimum_block is not None and min(run_lengths) < minimum_block:
        raise ValueError(f"selected run shorter than {minimum_block}: {min(run_lengths)}")
    valid_40_starts = sum(max(0, length - 40 + 1) for length in run_lengths)
    if valid_40_starts < args.minimum_valid_40_step_starts:
        raise ValueError(
            f"only {valid_40_starts} valid 40-step starts; "
            f"require {args.minimum_valid_40_step_starts}"
        )
    adjacent = (episode[1:] == episode[:-1]) & (timestep[1:] == timestep[:-1] + 1)
    states = tensors["states"].reshape(n, 45)
    next_states = tensors["next_states"].reshape(n, 45)
    state_link_error = float((next_states[:-1][adjacent] - states[1:][adjacent]).abs().max()) if bool(adjacent.any()) else 0.0
    next_observations = tensors["next_observations"].reshape(n, 45)
    action_alignment_error = float((next_observations[:, -12:] - actions).abs().max())
    if state_link_error > 1e-5:
        raise ValueError(f"next-state linkage error is {state_link_error}")
    if action_alignment_error > 1e-3:
        raise ValueError(f"next observation/action alignment error is {action_alignment_error}")
    collector_counts = {}
    if "collector_types" in data:
        collector = stack(data["collector_types"], "collector_types").reshape(n).long()
        collector_counts = {str(int(key)): int((collector == key).sum()) for key in collector.unique()}
        if args.require_expert and set(collector_counts) != {"1"}:
            raise ValueError(f"non-expert collector IDs present: {collector_counts}")
    expected_mode_counts = {
        "stand": 2000, "pure_x": 6250, "pure_y": 2500, "pure_yaw": 2000,
        "xy": 3500, "x_yaw": 4250, "y_yaw": 1250, "xy_yaw": 3250,
    }
    mode_names = {
        (False, False, False): "stand", (True, False, False): "pure_x",
        (False, True, False): "pure_y", (False, False, True): "pure_yaw",
        (True, True, False): "xy", (True, False, True): "x_yaw",
        (False, True, True): "y_yaw", (True, True, True): "xy_yaw",
    }
    commands = stack(data["commands"], "commands").reshape(n, 3)
    mode_counts = Counter(
        mode_names[tuple(bool(value) for value in row)]
        for row in (commands.abs() > 1e-3).tolist()
    )
    if dict(mode_counts) != expected_mode_counts:
        raise ValueError(f"actual command mode counts mismatch: {mode_counts}")
    contacts = tensors["contacts"].reshape(n, 4)
    termination_count = int((tensors["terminations"] > 0.5).sum())
    metadata = data.get("metadata", {})
    required_provenance = ("condition_id", "selection_source_sha256")
    missing = [key for key in required_provenance if not metadata.get(key)]
    if missing:
        raise ValueError(f"missing provenance metadata: {missing}")
    report = {
        "schema": "go2_matched_dataset_validation_v2",
        "dataset": str(dataset_path.resolve()),
        "transitions": n,
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
        "blocks": len(run_lengths),
        "block_length_min": min(run_lengths),
        "block_length_max": max(run_lengths),
        "valid_40_step_starts": valid_40_starts,
        "mode_counts": dict(mode_counts),
        "collector_counts": collector_counts,
        "action_min": actions.min(0).values.tolist(),
        "action_max": actions.max(0).values.tolist(),
        "action_saturation_fraction": float((actions.abs() >= 0.999).float().mean()),
        "next_state_link_max_error": state_link_error,
        "next_observation_action_max_error": action_alignment_error,
        "termination_count": termination_count,
        "contact_fraction_per_foot": contacts.mean(0).tolist(),
        "all_four_contact_fraction": float((contacts.sum(1) == 4).float().mean()),
        "metadata": metadata,
        "status": "pass",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "transitions", "blocks", "valid_40_step_starts",
        "next_state_link_max_error", "next_observation_action_max_error",
    )}, indent=2))


if __name__ == "__main__":
    main()
