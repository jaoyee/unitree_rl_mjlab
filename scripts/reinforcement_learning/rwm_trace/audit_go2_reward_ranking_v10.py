"""Audit whether the shared Go2 task reward prefers command-following trajectories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tasks.rwm_velocity.mdp.rewards import Go2RWMRewardState, compute_go2_imagination_reward


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates")
    source.add_argument("--summaries")
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-dt", type=float, default=0.02)
    return parser.parse_args()


def _trajectories(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("trajectories", "candidates", "rollouts"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if all(key in payload for key in ("states", "actions", "next_states", "trace_valid_masks")):
            from scripts.reinforcement_learning.rwm_trace.build_trace_replay_v10 import _candidate_windows
            from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key

            time_steps = int(stack_time_key(payload, "states").shape[0])
            return _candidate_windows(
                payload,
                length=time_steps,
                stride=time_steps,
                include_terminal_prefixes=True,
                minimum_terminal_length=1,
            )
    keys = sorted(map(str, payload.keys())) if isinstance(payload, dict) else None
    raise TypeError(f"Unsupported candidate payload: {type(payload)!r}, keys={keys}")


def _tensor(item: dict[str, Any], *keys: str) -> torch.Tensor:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return torch.as_tensor(value).float()
    raise KeyError(f"Missing all keys {keys}")


def _mode(item: dict[str, Any], command: torch.Tensor) -> str:
    for key in ("command_mode", "mode", "command_mode_name"):
        value = item.get(key)
        if value is not None:
            return str(value)
    active = [abs(float(value)) > 1e-5 for value in command.mean(dim=0)]
    names = {
        (False, False, False): "stand",
        (True, False, False): "pure_x",
        (False, True, False): "pure_y",
        (False, False, True): "pure_yaw",
        (True, True, False): "xy",
        (True, False, True): "x_yaw",
        (False, True, True): "y_yaw",
        (True, True, True): "xy_yaw",
    }
    return names[tuple(active)]


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    xr = torch.as_tensor(x).argsort().argsort().float().numpy()
    yr = torch.as_tensor(y).argsort().argsort().float().numpy()
    value = float(np.corrcoef(xr, yr)[0, 1])
    return value if math.isfinite(value) else None


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = _parse_args()
    if args.summaries:
        summaries = [json.loads(line) for line in Path(args.summaries).read_text(encoding="utf-8").splitlines()]
        rows: list[dict[str, float | str]] = []
        for summary in summaries:
            realization_values = []
            if summary.get("command_linear_active"):
                realization_values.append(float(summary["linear_velocity_realization_ratio_mean"]))
            if summary.get("command_yaw_active"):
                realization_values.append(float(summary["yaw_velocity_realization_ratio_mean"]))
            rows.append(
                {
                    "mode": str(summary["command_mode"]),
                    "reward": float(summary["reward_mean"]),
                    "return": float(summary["simulator_return"]),
                    "realization": float(np.mean(realization_values)) if realization_values else 1.0,
                    "direction_correct": 1.0 - float(summary["command_direction_violation_rate"]),
                    "speed": float(summary["base_speed_mean"]),
                    "xy_error": float(summary["linear_tracking_error_mean"]),
                    "yaw_error": float(summary["yaw_tracking_error_mean"]),
                    "saturation": float(summary["action_saturation_fraction"]),
                    "selected": float(bool(summary.get("selected", False))),
                }
            )
    else:
        rows = []
        assert args.candidates is not None
        payload = torch.load(args.candidates, map_location="cpu", weights_only=False)
        candidates = _trajectories(payload)
    weighted_names = (
        "track_linear_velocity",
        "track_angular_velocity",
        "body_orientation_l2",
        "body_ang_vel",
        "dof_torques_l2",
        "dof_acc_l2",
        "action_rate_l2",
        "foot_gait",
        "stand_still",
    )

    for item in candidates if not args.summaries else []:
        states = _tensor(item, "states", "observations")
        next_states = _tensor(item, "next_states", "next_observations")
        actions = _tensor(item, "actions")
        commands = _tensor(item, "commands")
        contacts = _tensor(item, "contacts", "foot_contacts")
        previous = item.get("prev_actions")
        previous_actions = torch.as_tensor(previous).float() if previous is not None else torch.zeros_like(actions)
        length = min(len(states), len(next_states), len(actions), len(commands), len(contacts))
        reward_state = Go2RWMRewardState.create(1, actions.shape[-1], "cpu", args.step_dt)
        reward_state.last_joint_vel.copy_(states[0, 21:33].view(1, -1))
        reward_state.last_action.copy_(previous_actions[0].view(1, -1))
        reward_values: list[float] = []
        term_values: dict[str, list[float]] = defaultdict(list)
        for step in range(length):
            reward, terms = compute_go2_imagination_reward(
                next_states[step].view(1, -1),
                actions[step].view(1, -1),
                commands[step].view(1, -1),
                contacts[step].view(1, -1),
                torch.tensor([step]),
                reward_state,
                torch.zeros(1),
            )
            reward_values.append(float(reward.item()))
            for name in weighted_names:
                weight = float(getattr(reward_state.weights, name))
                term_values[name].append(weight * float(terms[name].item()) * args.step_dt)

        command = commands[:length]
        velocity = next_states[:length, 0:3]
        angular = next_states[:length, 3:6]
        linear_cmd = command[:, :2]
        linear_norm = torch.linalg.norm(linear_cmd, dim=1)
        linear_projection = (velocity[:, :2] * linear_cmd).sum(dim=1) / linear_norm.clamp_min(1e-6)
        linear_realization = linear_projection / linear_norm.clamp_min(1e-6)
        yaw_realization = angular[:, 2] * command[:, 2] / command[:, 2].square().clamp_min(1e-6)
        active_linear = linear_norm > 1e-5
        active_yaw = command[:, 2].abs() > 1e-5
        realization_parts = []
        if active_linear.any():
            realization_parts.append(float(linear_realization[active_linear].mean()))
        if active_yaw.any():
            realization_parts.append(float(yaw_realization[active_yaw].mean()))
        realization = float(np.mean(realization_parts)) if realization_parts else 1.0
        direction_ok = []
        if active_linear.any():
            direction_ok.append(float((linear_projection[active_linear] > 0).float().mean()))
        if active_yaw.any():
            direction_ok.append(float(((angular[:, 2] * command[:, 2])[active_yaw] > 0).float().mean()))
        row: dict[str, float | str] = {
            "mode": _mode(item, command),
            "reward": float(np.mean(reward_values)),
            "return": float(np.sum(reward_values)),
            "realization": realization,
            "direction_correct": float(np.mean(direction_ok)) if direction_ok else 1.0,
            "speed": float(torch.linalg.norm(velocity[:, :2], dim=1).mean()),
            "xy_error": float(torch.linalg.norm(velocity[:, :2] - linear_cmd, dim=1).mean()),
            "yaw_error": float((angular[:, 2] - command[:, 2]).abs().mean()),
            "saturation": float((actions.abs() >= 0.95).any(dim=1).float().mean()),
        }
        for name in weighted_names:
            row[name] = float(np.mean(term_values[name]))
        rows.append(row)

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append({key: float(value) for key, value in row.items() if key != "mode"})

    report: dict[str, Any] = {"candidate_count": len(rows), "modes": {}}
    for mode, mode_rows in sorted(grouped.items()):
        ordered = sorted(mode_rows, key=lambda row: row["reward"])
        count = max(1, len(ordered) // 4)
        bottom, top = ordered[:count], ordered[-count:]
        report["modes"][mode] = {
            "count": len(mode_rows),
            "spearman_reward_realization": _spearman(
                [row["reward"] for row in mode_rows], [row["realization"] for row in mode_rows]
            ),
            "spearman_reward_xy_error": _spearman(
                [row["reward"] for row in mode_rows], [row["xy_error"] for row in mode_rows]
            ),
            "spearman_reward_yaw_error": _spearman(
                [row["reward"] for row in mode_rows], [row["yaw_error"] for row in mode_rows]
            ),
            "all": {key: _mean(mode_rows, key) for key in mode_rows[0]},
            "top_reward_quartile": {key: _mean(top, key) for key in top[0]},
            "bottom_reward_quartile": {key: _mean(bottom, key) for key in bottom[0]},
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} for {len(rows)} candidates")


if __name__ == "__main__":
    main()
