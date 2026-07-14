"""Validate the exact 25k expert-only subset used by aligned sim experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import (
    COLLECTOR_NAME_TO_ID,
    load_mixed_dataset,
)


TARGET_MODE_WEIGHTS = {
    "stand": 0.08,
    "pure_x": 0.25,
    "pure_y": 0.10,
    "pure_yaw": 0.08,
    "xy": 0.14,
    "x_yaw": 0.17,
    "y_yaw": 0.05,
    "xy_yaw": 0.13,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_transitions", type=int, default=25_000)
    parser.add_argument("--mode_tolerance", type=float, default=0.08)
    return parser.parse_args()


def _stack(dataset: dict, key: str) -> torch.Tensor:
    value = dataset[key]
    return value if isinstance(value, torch.Tensor) else torch.stack(value, dim=0)


def _command_mode_counts(commands: torch.Tensor) -> dict[str, int]:
    active = commands.abs() > 1.0e-6
    x, y, yaw = (active[..., index] for index in range(3))
    masks = {
        "stand": ~x & ~y & ~yaw,
        "pure_x": x & ~y & ~yaw,
        "pure_y": ~x & y & ~yaw,
        "pure_yaw": ~x & ~y & yaw,
        "xy": x & y & ~yaw,
        "x_yaw": x & ~y & yaw,
        "y_yaw": ~x & y & yaw,
        "xy_yaw": x & y & yaw,
    }
    return {name: int(mask.sum().item()) for name, mask in masks.items()}


def main() -> None:
    args = _parse_args()
    dataset = load_mixed_dataset(args.dataset_path)
    commands = _stack(dataset, "commands").float()
    collectors = _stack(dataset, "collector_types").long()
    time_steps, num_envs = collectors.shape
    transitions = time_steps * num_envs
    if transitions != int(args.expected_transitions):
        raise ValueError(f"Expected {args.expected_transitions} transitions, got {transitions}.")
    expert_id = COLLECTOR_NAME_TO_ID["expert"]
    if not bool((collectors == expert_id).all()):
        values, counts = collectors.unique(return_counts=True)
        raise ValueError(f"Subset is not expert-only: {dict(zip(values.tolist(), counts.tolist()))}")
    counts = _command_mode_counts(commands)
    proportions = {name: count / transitions for name, count in counts.items()}
    deviations = {
        name: abs(proportions[name] - target)
        for name, target in TARGET_MODE_WEIGHTS.items()
    }
    if max(deviations.values()) > float(args.mode_tolerance):
        raise ValueError(
            f"Command distribution exceeds tolerance {args.mode_tolerance}: "
            f"proportions={proportions}, deviations={deviations}"
        )
    report = {
        "status": "passed",
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "time_steps": time_steps,
        "num_envs": num_envs,
        "num_transitions": transitions,
        "collector": "expert",
        "command_mode_counts": counts,
        "command_mode_proportions": proportions,
        "target_mode_weights": TARGET_MODE_WEIGHTS,
        "absolute_deviations": deviations,
        "metadata": dataset.get("metadata") or {},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "num_transitions", "collector", "command_mode_proportions")}, indent=2))


if __name__ == "__main__":
    main()
