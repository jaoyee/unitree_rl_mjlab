#!/usr/bin/env python3
"""Summarize contiguous expert-dataset windows for initial TRACE scorer labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.trajectory import summarize_go2_trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window_length", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--action_saturation_threshold", type=float, default=0.95)
    return parser.parse_args()


def _flat(data: dict[str, Any], key: str, width: int | None = None) -> torch.Tensor:
    value = stack_time_key(data, key).detach().cpu()
    if width is None:
        return value.reshape(-1)
    return value.reshape(-1, width)


def main() -> None:
    args = _parse_args()
    if args.window_length < 2 or args.stride < 1:
        raise ValueError("window_length must be >=2 and stride must be positive")
    path = Path(args.dataset).expanduser().resolve()
    data = torch.load(path, map_location="cpu", weights_only=False)
    states = _flat(data, "states", 45)
    actions = _flat(data, "actions", 12)
    next_states = _flat(data, "next_states", 45)
    contacts = _flat(data, "contacts", 4)
    terminations = _flat(data, "terminations", 1)
    commands = _flat(data, "commands", 3)
    rewards = _flat(data, "rewards")
    prev_actions = _flat(data, "prev_actions", 12)
    episode_ids = _flat(data, "episode_ids").long()
    timesteps = _flat(data, "timesteps").long()
    count = len(states)
    if any(len(value) != count for value in (
        actions, next_states, contacts, terminations, commands, rewards,
        prev_actions, episode_ids, timesteps,
    )):
        raise ValueError("Dataset transition fields have inconsistent flattened lengths.")

    namespace = sha256_path(path)[:16]
    summaries: list[dict[str, Any]] = []
    start = 0
    while start < count:
        stop = start + 1
        while (
            stop < count
            and episode_ids[stop] == episode_ids[stop - 1]
            and timesteps[stop] == timesteps[stop - 1] + 1
        ):
            stop += 1
        episode = int(episode_ids[start])
        for window_start in range(start, stop - args.window_length + 1, args.stride):
            window_stop = window_start + args.window_length
            trajectory = {
                "states": states[window_start:window_stop],
                "actions": actions[window_start:window_stop],
                "next_states": next_states[window_start:window_stop],
                "contacts": contacts[window_start:window_stop],
                "terminations": terminations[window_start:window_stop].reshape(-1).bool(),
                "commands": commands[window_start:window_stop],
                "rewards": rewards[window_start:window_stop],
                "prev_actions": prev_actions[window_start:window_stop],
                "trajectory_id": f"dataset_ep{episode:06d}_t{int(timesteps[window_start]):06d}",
                "start_state_id": window_start,
                "reset_reconstruction_error": float("nan"),
                "simulator_mismatch": {"source": "expert_dataset"},
            }
            summary = summarize_go2_trajectory(trajectory)
            saturated = (actions[window_start:window_stop].abs() > args.action_saturation_threshold).any(dim=-1)
            summary["action_saturation_fraction"] = float(saturated.float().mean())
            summary["action_abs_mean"] = float(actions[window_start:window_stop].abs().mean())
            summary["action_delta_abs_mean"] = float(
                (actions[window_start:window_stop] - prev_actions[window_start:window_stop]).abs().mean()
            )
            summary["start_state_key"] = f"{namespace}:{window_start}"
            summary["comparison_group_key"] = f"{namespace}:episode:{episode}"
            summary["candidate_namespace"] = namespace
            summary["source_kind"] = "expert_dataset_window"
            summaries.append(summary)
        start = stop

    if len(summaries) < 2:
        raise RuntimeError("Dataset produced fewer than two contiguous scorer windows.")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps({
        "dataset": str(path),
        "dataset_sha256": sha256_path(path),
        "window_length": args.window_length,
        "stride": args.stride,
        "summary_count": len(summaries),
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
