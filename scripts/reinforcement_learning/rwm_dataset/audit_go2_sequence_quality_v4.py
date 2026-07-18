#!/usr/bin/env python3
"""Audit termination balance and contiguous sequence support in a Go2 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=(40, 100, 200))
    parser.add_argument("--zero-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--action-saturation-threshold", type=float, default=0.95)
    return parser.parse_args()


def _time_env(dataset: dict[str, Any], key: str) -> torch.Tensor | None:
    if dataset.get(key) is None:
        return None
    value = stack_time_key(dataset, key)
    if value.ndim == 1:
        value = value[:, None]
    return value


def _mode_ids(commands: np.ndarray, epsilon: float) -> np.ndarray:
    active = np.abs(commands) > float(epsilon)
    result = np.full(active.shape[:-1], -1, dtype=np.int8)
    for bits, mode_id in MODE_IDS.items():
        result[np.all(active == np.asarray(bits), axis=-1)] = mode_id
    return result


def _describe(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "min": 0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0, "mean": 0.0}
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def audit(dataset: dict[str, Any], sequence_lengths: list[int], epsilon: float, saturation: float) -> dict[str, Any]:
    states = _time_env(dataset, "states")
    actions = _time_env(dataset, "actions")
    commands = _time_env(dataset, "commands")
    terminations = _time_env(dataset, "terminations")
    episode_ids = _time_env(dataset, "episode_ids")
    timesteps = _time_env(dataset, "timesteps")
    if states is None or actions is None or commands is None or terminations is None:
        raise ValueError("Dataset must contain states, actions, commands, and terminations.")
    time_steps, num_envs = states.shape[:2]
    term = terminations.reshape(time_steps, num_envs, -1).bool().any(dim=-1).cpu().numpy()
    if episode_ids is None:
        episode_ids = torch.arange(num_envs)[None, :].expand(time_steps, -1)
    else:
        episode_ids = episode_ids.reshape(time_steps, num_envs, -1)[..., 0]
    if timesteps is None:
        timesteps = torch.arange(time_steps)[:, None].expand(-1, num_envs)
    else:
        timesteps = timesteps.reshape(time_steps, num_envs, -1)[..., 0]
    episode_ids_np = episode_ids.cpu().numpy()
    timesteps_np = timesteps.cpu().numpy()

    run_lengths: list[int] = []
    terminal_run_lengths: list[int] = []
    positive_by_run = 0
    for env in range(num_envs):
        start = 0
        for step in range(1, time_steps + 1):
            contiguous = (
                step < time_steps
                and episode_ids_np[step, env] == episode_ids_np[step - 1, env]
                and timesteps_np[step, env] == timesteps_np[step - 1, env] + 1
            )
            if contiguous:
                continue
            length = step - start
            if length > 0:
                run_lengths.append(length)
                if term[start:step, env].any():
                    terminal_run_lengths.append(length)
                    positive_by_run += 1
            start = step

    run_lengths_np = np.asarray(run_lengths, dtype=np.int64)
    command_np = commands.detach().cpu().numpy()
    mode_ids = _mode_ids(command_np, epsilon)
    mode_counts = {
        MODE_NAMES[mode_id]: int((mode_ids == mode_id).sum())
        for mode_id in range(len(MODE_NAMES))
    }
    action_np = actions.detach().cpu().numpy()
    saturated = (np.abs(action_np) > float(saturation)).any(axis=-1)
    finite = bool(
        torch.isfinite(states).all()
        and torch.isfinite(actions).all()
        and torch.isfinite(commands).all()
    )
    source_metadata = dict((dataset.get("metadata") or {}))
    report = {
        "schema": "go2_sequence_quality_v4",
        "time_steps": int(time_steps),
        "num_envs": int(num_envs),
        "num_transitions": int(time_steps * num_envs),
        "state_dim": int(states.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "all_finite": finite,
        "termination_positive_transitions": int(term.sum()),
        "termination_positive_fraction": float(term.mean()),
        "terminal_contiguous_runs": int(positive_by_run),
        "contiguous_runs": _describe(run_lengths_np),
        "terminal_run_lengths": _describe(np.asarray(terminal_run_lengths, dtype=np.int64)),
        "sequence_support": {
            str(length): {
                "eligible_runs": int((run_lengths_np >= length).sum()),
                "eligible_transitions": int(run_lengths_np[run_lengths_np >= length].sum()),
                "valid_starts": int(np.maximum(run_lengths_np - length + 1, 0).sum()),
            }
            for length in sequence_lengths
        },
        "command_mode_counts": mode_counts,
        "action_abs_mean": float(np.abs(action_np).mean()),
        "action_abs_max": float(np.abs(action_np).max()),
        "action_saturation_transition_fraction": float(saturated.mean()),
        "metadata_condition_id": source_metadata.get("condition_id"),
        "metadata_selection_kind": source_metadata.get("selection_kind"),
    }
    return report


def main() -> None:
    args = parse_args()
    if any(length < 1 for length in args.sequence_lengths):
        raise ValueError("Sequence lengths must be positive.")
    input_path = Path(args.input).expanduser().resolve()
    print(f"Loading {input_path}", flush=True)
    dataset = torch.load(input_path, map_location="cpu", weights_only=False)
    report = audit(
        dataset,
        sorted(set(args.sequence_lengths)),
        args.zero_epsilon,
        args.action_saturation_threshold,
    )
    report["input"] = str(input_path)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
