"""Trajectory extraction and Go2-specific summary statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trend(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan")
    return float(np.polyfit(np.arange(len(values), dtype=np.float64), values, 1)[0])


def summarize_go2_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Summarize one `[time, feature]` candidate without using privileged IDs."""

    states = _as_numpy(trajectory["states"]).astype(np.float64)
    next_states = _as_numpy(trajectory["next_states"]).astype(np.float64)
    actions = _as_numpy(trajectory["actions"]).astype(np.float64)
    rewards = _as_numpy(trajectory["rewards"]).astype(np.float64).reshape(-1)
    commands = _as_numpy(trajectory["commands"]).astype(np.float64)
    contacts = _as_numpy(trajectory["contacts"]).astype(np.float64)
    terminations = _as_numpy(trajectory["terminations"]).astype(bool).reshape(-1)

    if not (len(states) == len(next_states) == len(actions) == len(rewards) == len(commands)):
        raise ValueError("Candidate trajectory arrays must have the same time dimension.")
    state_delta = next_states - states
    action_norm = np.linalg.norm(actions, axis=-1)
    action_delta = np.diff(actions, axis=0)
    gravity = next_states[:, 6:9]
    gravity_norm = np.maximum(np.linalg.norm(gravity, axis=-1), 1.0e-8)
    tilt = np.arccos(np.clip(-gravity[:, 2] / gravity_norm, -1.0, 1.0))
    linear_error = np.linalg.norm(next_states[:, 0:2] - commands[:, 0:2], axis=-1)
    yaw_error = np.abs(next_states[:, 5] - commands[:, 2])
    contact_switches = np.abs(np.diff(contacts, axis=0)) if len(contacts) > 1 else np.zeros((0, 4))
    finite = all(np.isfinite(value).all() for value in (states, next_states, actions, rewards, commands, contacts))

    return {
        "trajectory_id": str(trajectory.get("trajectory_id", "")),
        "start_state_id": int(trajectory.get("start_state_id", -1)),
        "simulator_return": float(rewards.sum()),
        "dataset_return": (
            float(_as_numpy(trajectory["dataset_rewards"]).sum())
            if trajectory.get("dataset_rewards") is not None
            else float("nan")
        ),
        "survival_length": int(len(rewards)),
        "terminal_flag": bool(terminations.any()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_trend_slope": _trend(rewards),
        "action_norm_mean": float(action_norm.mean()),
        "action_norm_std": float(action_norm.std()),
        "action_saturation_rate": float((np.abs(actions) >= 0.98).mean()),
        "action_delta_norm_mean": float(np.linalg.norm(action_delta, axis=-1).mean()) if len(action_delta) else 0.0,
        "state_delta_norm_mean": float(np.linalg.norm(state_delta, axis=-1).mean()),
        "base_speed_mean": float(np.linalg.norm(next_states[:, 0:2], axis=-1).mean()),
        "linear_tracking_error_mean": float(linear_error.mean()),
        "yaw_tracking_error_mean": float(yaw_error.mean()),
        "tilt_mean": float(tilt.mean()),
        "tilt_max": float(tilt.max()),
        "joint_velocity_rms": float(np.sqrt(np.mean(np.square(next_states[:, 21:33])))),
        "actuator_force_rms": float(np.sqrt(np.mean(np.square(next_states[:, 33:45])))),
        "contact_fraction_mean": float(contacts.mean()),
        "contact_switch_rate": float(contact_switches.mean()) if contact_switches.size else 0.0,
        "nonfinite_flag": not finite,
        "reset_reconstruction_error": float(trajectory.get("reset_reconstruction_error", float("nan"))),
        "simulator_mismatch": dict(trajectory.get("simulator_mismatch") or {}),
    }
