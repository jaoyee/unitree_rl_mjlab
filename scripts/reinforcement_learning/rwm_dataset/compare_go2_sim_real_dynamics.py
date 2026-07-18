#!/usr/bin/env python3
"""Compare sim/real Go2 datasets beyond command-count matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


BLOCKS = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "joint_pos_rel": slice(9, 21),
    "joint_vel": slice(21, 33),
    "actuator_force": slice(33, 45),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sim-dataset", required=True)
    parser.add_argument("--real-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-length", type=int, default=40)
    return parser.parse_args()


def stack(value: Any) -> torch.Tensor:
    if isinstance(value, list):
        return torch.stack([torch.as_tensor(row) for row in value])
    return torch.as_tensor(value)


def flatten_time_env(value: Any) -> np.ndarray:
    tensor = stack(value).detach().cpu()
    if tensor.ndim >= 3:
        tensor = tensor.reshape(-1, *tensor.shape[2:])
    return tensor.numpy()


def effective_starts(dataset: dict[str, Any], history: int) -> int:
    episodes = stack(dataset["episode_ids"]).reshape(stack(dataset["episode_ids"]).shape[:2])
    timesteps = stack(dataset["timesteps"]).reshape(stack(dataset["timesteps"]).shape[:2])
    total = 0
    time_steps, num_envs = episodes.shape
    for env_id in range(num_envs):
        ids = episodes[:, env_id]
        ts = timesteps[:, env_id]
        if time_steps < history:
            continue
        same_episode = torch.ones(time_steps - history + 1, dtype=torch.bool)
        consecutive = torch.ones_like(same_episode)
        for offset in range(1, history):
            same_episode &= ids[offset : offset + len(same_episode)] == ids[: len(same_episode)]
            consecutive &= ts[offset : offset + len(consecutive)] == ts[: len(consecutive)] + offset
        total += int((same_episode & consecutive).sum())
    return total


def scalar_stats(values: np.ndarray) -> dict[str, float | list[float]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    quantiles = np.quantile(finite, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "abs_mean": float(np.abs(finite).mean()),
        "quantiles": quantiles.tolist(),
        "finite_fraction": float(len(finite) / max(1, len(values))),
    }


def quantile_distance(left: np.ndarray, right: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 1001)
    left_q = np.quantile(np.asarray(left).reshape(-1), grid)
    right_q = np.quantile(np.asarray(right).reshape(-1), grid)
    return float(np.mean(np.abs(left_q - right_q)))


def summarize(dataset: dict[str, Any], history: int) -> dict[str, Any]:
    states = flatten_time_env(dataset["states"])
    next_states = flatten_time_env(dataset["next_states"])
    actions = flatten_time_env(dataset["actions"])
    contacts = flatten_time_env(dataset["contacts"])
    commands = flatten_time_env(dataset["commands"])
    delta = next_states - states
    result: dict[str, Any] = {
        "transitions": int(len(states)),
        "effective_history_starts": effective_starts(dataset, history),
        "action": scalar_stats(actions),
        "action_saturation_rate": float((np.abs(actions) >= 0.98).mean()),
        "contact_fraction": float(contacts.mean()),
        "commands": {
            "x": scalar_stats(commands[:, 0]),
            "y": scalar_stats(commands[:, 1]),
            "yaw": scalar_stats(commands[:, 2]),
        },
        "blocks": {},
    }
    for name, block in BLOCKS.items():
        result["blocks"][name] = {
            "state": scalar_stats(states[:, block]),
            "delta": scalar_stats(delta[:, block]),
            "delta_l2_mean": float(np.linalg.norm(delta[:, block], axis=-1).mean()),
        }
    return result


def main() -> None:
    args = parse_args()
    sim = torch.load(args.sim_dataset, map_location="cpu", weights_only=False)
    real = torch.load(args.real_dataset, map_location="cpu", weights_only=False)
    sim_summary = summarize(sim, args.history_length)
    real_summary = summarize(real, args.history_length)
    sim_states = flatten_time_env(sim["states"])
    real_states = flatten_time_env(real["states"])
    sim_next = flatten_time_env(sim["next_states"])
    real_next = flatten_time_env(real["next_states"])
    distances = {}
    for name, block in BLOCKS.items():
        distances[name] = {
            "state_quantile_l1": quantile_distance(sim_states[:, block], real_states[:, block]),
            "delta_quantile_l1": quantile_distance(
                sim_next[:, block] - sim_states[:, block],
                real_next[:, block] - real_states[:, block],
            ),
            "real_to_sim_delta_l2_ratio": (
                real_summary["blocks"][name]["delta_l2_mean"]
                / max(sim_summary["blocks"][name]["delta_l2_mean"], 1.0e-12)
            ),
        }
    payload = {
        "format_version": "go2_sim_real_dynamics_alignment_report_v1",
        "sim_dataset": str(Path(args.sim_dataset).resolve()),
        "real_dataset": str(Path(args.real_dataset).resolve()),
        "history_length": int(args.history_length),
        "sim": sim_summary,
        "real": real_summary,
        "sim_real_distances": distances,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
