"""Trajectory extraction and Go2-specific summary statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


GO2_COMMAND_MODES = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")


def classify_go2_command(command: np.ndarray, *, eps: float = 1.0e-4) -> str:
    """Classify one `[vx, vy, yaw]` command without using magnitude-specific thresholds."""

    vx, vy, yaw = np.abs(np.asarray(command, dtype=np.float64).reshape(-1)[:3]) > eps
    if not (vx or vy or yaw):
        return "stand"
    if vx and not vy and not yaw:
        return "pure_x"
    if vy and not vx and not yaw:
        return "pure_y"
    if yaw and not vx and not vy:
        return "pure_yaw"
    if vx and vy and not yaw:
        return "xy"
    if vx and yaw and not vy:
        return "x_yaw"
    if vy and yaw and not vx:
        return "y_yaw"
    return "xy_yaw"


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trend(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan")
    return float(np.polyfit(np.arange(len(values), dtype=np.float64), values, 1)[0])


def _longest_boolean_run(values: np.ndarray, target: bool) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if bool(value) is target else 0
        longest = max(longest, current)
    return int(longest)


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
    command_mean = commands[:, :3].mean(axis=0)
    command_mode = classify_go2_command(command_mean)
    command_linear = commands[:, 0:2]
    command_linear_norm = np.linalg.norm(command_linear, axis=-1)
    linear_active = command_linear_norm > 1.0e-4
    linear_projection = np.zeros(len(commands), dtype=np.float64)
    linear_projection[linear_active] = (
        np.sum(next_states[linear_active, 0:2] * command_linear[linear_active], axis=-1)
        / command_linear_norm[linear_active]
    )
    linear_realization = np.full(len(commands), np.nan, dtype=np.float64)
    linear_realization[linear_active] = linear_projection[linear_active] / command_linear_norm[linear_active]
    yaw_active = np.abs(commands[:, 2]) > 1.0e-4
    yaw_realization = np.full(len(commands), np.nan, dtype=np.float64)
    yaw_realization[yaw_active] = next_states[yaw_active, 5] / commands[yaw_active, 2]
    direction_checks = []
    if linear_active.any():
        direction_checks.append(linear_projection[linear_active] <= 0.0)
    if yaw_active.any():
        direction_checks.append(yaw_realization[yaw_active] <= 0.0)
    direction_violation_rate = (
        float(np.concatenate(direction_checks).mean()) if direction_checks else 0.0
    )
    step_dt = float(trajectory.get("step_dt", 0.02))
    nominal_horizon = int(trajectory.get("nominal_horizon", len(rewards)))
    if nominal_horizon < len(rewards) or nominal_horizon <= 0 or step_dt <= 0.0:
        raise ValueError("Trajectory nominal_horizon and step_dt must cover the observed trajectory.")
    duration = max(len(rewards) * step_dt, step_dt)
    steady_start = len(commands) // 2
    steady_linear = linear_active[steady_start:]
    steady_yaw = yaw_active[steady_start:]
    projected_displacement = float(linear_projection.sum() * step_dt)
    expected_linear_displacement = float(command_linear_norm.sum() * step_dt)
    yaw_displacement = float(next_states[:, 5].sum() * step_dt)
    expected_yaw_displacement = float(commands[:, 2].sum() * step_dt)
    contact_switches = np.abs(np.diff(contacts, axis=0)) if len(contacts) > 1 else np.zeros((0, 4))
    contact_mean = contacts.mean(axis=0) if contacts.ndim == 2 and contacts.shape[-1] >= 4 else np.full(4, np.nan)
    contact_switch_mean = (
        contact_switches.mean(axis=0)
        if contact_switches.ndim == 2 and contact_switches.shape[-1] >= 4
        else np.full(4, np.nan)
    )
    finite = all(np.isfinite(value).all() for value in (states, next_states, actions, rewards, commands, contacts))
    swing_counts = (
        np.sum((contacts[:-1, :4] > 0.5) & (contacts[1:, :4] <= 0.5), axis=0)
        if len(contacts) > 1
        else np.zeros(4, dtype=np.int64)
    )
    longest_stance = np.asarray([_longest_boolean_run(contacts[:, i] > 0.5, True) for i in range(4)])
    longest_swing = np.asarray([_longest_boolean_run(contacts[:, i] > 0.5, False) for i in range(4)])

    base_positions = trajectory.get("trace_base_positions_w")
    if base_positions is not None:
        base_positions = _as_numpy(base_positions).astype(np.float64)
        base_net_displacement = base_positions[-1] - base_positions[0]
        base_height_mean = float(base_positions[:, 2].mean())
        base_height_std = float(base_positions[:, 2].std())
        base_height_trend = _trend(base_positions[:, 2])
    else:
        base_net_displacement = np.full(3, np.nan)
        base_height_mean = base_height_std = base_height_trend = float("nan")

    foot_positions = trajectory.get("trace_foot_positions_w")
    foot_velocities = trajectory.get("trace_foot_velocities_w")
    if foot_positions is not None and foot_velocities is not None:
        foot_positions = _as_numpy(foot_positions).astype(np.float64)
        foot_velocities = _as_numpy(foot_velocities).astype(np.float64)
        foot_height_max = foot_positions[:, :4, 2].max(axis=0)
        foot_height_range = np.ptp(foot_positions[:, :4, 2], axis=0)
        foot_speed_mean = np.linalg.norm(foot_velocities[:, :4], axis=-1).mean(axis=0)
    else:
        foot_height_max = np.full(4, np.nan)
        foot_height_range = np.full(4, np.nan)
        foot_speed_mean = np.full(4, np.nan)

    return {
        "trajectory_id": str(trajectory.get("trajectory_id", "")),
        "start_state_id": int(trajectory.get("start_state_id", -1)),
        "simulator_return": float(rewards.sum()),
        "simulator_return_per_step": float(rewards.mean()),
        "dataset_return": (
            float(_as_numpy(trajectory["dataset_rewards"]).sum())
            if trajectory.get("dataset_rewards") is not None
            else float("nan")
        ),
        "survival_length": int(len(rewards)),
        "survival_fraction": float(len(rewards) / nominal_horizon),
        "terminal_flag": bool(terminations.any()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_trend_slope": _trend(rewards),
        "reward_trend_per_second": _trend(rewards) / step_dt,
        "action_norm_mean": float(action_norm.mean()),
        "action_norm_std": float(action_norm.std()),
        "action_saturation_rate": float((np.abs(actions) >= 0.98).mean()),
        "action_delta_norm_mean": float(np.linalg.norm(action_delta, axis=-1).mean()) if len(action_delta) else 0.0,
        "state_delta_norm_mean": float(np.linalg.norm(state_delta, axis=-1).mean()),
        "base_speed_mean": float(np.linalg.norm(next_states[:, 0:2], axis=-1).mean()),
        "command_vx_mean": float(command_mean[0]),
        "command_vy_mean": float(command_mean[1]),
        "command_yaw_mean": float(command_mean[2]),
        "command_linear_speed_mean": float(command_linear_norm.mean()),
        "command_linear_active": bool(linear_active.any()),
        "command_yaw_active": bool(yaw_active.any()),
        "command_mode": command_mode,
        **{f"command_mode_{mode}": float(command_mode == mode) for mode in GO2_COMMAND_MODES},
        "linear_velocity_projection_mean": (
            float(linear_projection[linear_active].mean()) if linear_active.any() else 0.0
        ),
        "linear_velocity_realization_ratio_mean": (
            float(np.nanmean(linear_realization)) if linear_active.any() else float("nan")
        ),
        "linear_velocity_realization_ratio_steady": (
            float(np.nanmean(linear_realization[steady_start:][steady_linear]))
            if steady_linear.any()
            else float("nan")
        ),
        "yaw_velocity_realization_ratio_mean": (
            float(np.nanmean(yaw_realization)) if yaw_active.any() else float("nan")
        ),
        "yaw_velocity_realization_ratio_steady": (
            float(np.nanmean(yaw_realization[steady_start:][steady_yaw]))
            if steady_yaw.any()
            else float("nan")
        ),
        "command_direction_violation_rate": direction_violation_rate,
        "command_direction_correct_fraction": 1.0 - direction_violation_rate,
        "command_projected_displacement": projected_displacement,
        "command_projected_velocity_window": projected_displacement / duration,
        "command_expected_linear_displacement": expected_linear_displacement,
        "command_displacement_realization_ratio": (
            projected_displacement / expected_linear_displacement
            if expected_linear_displacement > 1.0e-8
            else float("nan")
        ),
        "yaw_net_change": yaw_displacement,
        "yaw_velocity_window": yaw_displacement / duration,
        "yaw_expected_change": expected_yaw_displacement,
        "yaw_displacement_realization_ratio": (
            yaw_displacement / expected_yaw_displacement
            if abs(expected_yaw_displacement) > 1.0e-8
            else float("nan")
        ),
        "linear_tracking_error_mean": float(linear_error.mean()),
        "linear_tracking_error_steady": float(linear_error[steady_start:].mean()),
        "yaw_tracking_error_mean": float(yaw_error.mean()),
        "yaw_tracking_error_steady": float(yaw_error[steady_start:].mean()),
        "tilt_mean": float(tilt.mean()),
        "tilt_max": float(tilt.max()),
        "roll_pitch_rate_rms": float(np.sqrt(np.mean(np.square(next_states[:, 3:5])))),
        "joint_velocity_rms": float(np.sqrt(np.mean(np.square(next_states[:, 21:33])))),
        "actuator_force_rms": float(np.sqrt(np.mean(np.square(next_states[:, 33:45])))),
        "contact_fraction_mean": float(contacts.mean()),
        "contact_switch_rate": float(contact_switches.mean()) if contact_switches.size else 0.0,
        "contact_fraction_fr": float(contact_mean[0]),
        "contact_fraction_fl": float(contact_mean[1]),
        "contact_fraction_rr": float(contact_mean[2]),
        "contact_fraction_rl": float(contact_mean[3]),
        "contact_switch_rate_fr": float(contact_switch_mean[0]),
        "contact_switch_rate_fl": float(contact_switch_mean[1]),
        "contact_switch_rate_rr": float(contact_switch_mean[2]),
        "contact_switch_rate_rl": float(contact_switch_mean[3]),
        "foot_swing_count_fr": int(swing_counts[0]),
        "foot_swing_count_fl": int(swing_counts[1]),
        "foot_swing_count_rr": int(swing_counts[2]),
        "foot_swing_count_rl": int(swing_counts[3]),
        "foot_swing_rate_fr": float(swing_counts[0] / duration),
        "foot_swing_rate_fl": float(swing_counts[1] / duration),
        "foot_swing_rate_rr": float(swing_counts[2] / duration),
        "foot_swing_rate_rl": float(swing_counts[3] / duration),
        "longest_stance_steps_fr": int(longest_stance[0]),
        "longest_stance_steps_fl": int(longest_stance[1]),
        "longest_stance_steps_rr": int(longest_stance[2]),
        "longest_stance_steps_rl": int(longest_stance[3]),
        "longest_stance_fraction_rr": float(longest_stance[2] / len(rewards)),
        "longest_stance_fraction_rl": float(longest_stance[3] / len(rewards)),
        "longest_swing_steps_fr": int(longest_swing[0]),
        "longest_swing_steps_fl": int(longest_swing[1]),
        "longest_swing_steps_rr": int(longest_swing[2]),
        "longest_swing_steps_rl": int(longest_swing[3]),
        "rear_duty_factor_abs_difference": float(abs(contact_mean[2] - contact_mean[3])),
        # joint_pos_rel occupies state[9:21] in FR, FL, RL, RR order.
        # RR calf is the final joint, state index 20.
        "rr_calf_relative_position_mean": float(next_states[:, 20].mean()),
        "base_net_displacement_x": float(base_net_displacement[0]),
        "base_net_displacement_y": float(base_net_displacement[1]),
        "base_velocity_window_x": float(base_net_displacement[0] / duration),
        "base_velocity_window_y": float(base_net_displacement[1] / duration),
        "base_height_mean": base_height_mean,
        "base_height_std": base_height_std,
        "base_height_trend": base_height_trend,
        "base_height_trend_per_second": base_height_trend / step_dt,
        "foot_height_max_fr": float(foot_height_max[0]),
        "foot_height_max_fl": float(foot_height_max[1]),
        "foot_height_max_rr": float(foot_height_max[2]),
        "foot_height_max_rl": float(foot_height_max[3]),
        "foot_height_range_fr": float(foot_height_range[0]),
        "foot_height_range_fl": float(foot_height_range[1]),
        "foot_height_range_rr": float(foot_height_range[2]),
        "foot_height_range_rl": float(foot_height_range[3]),
        "foot_speed_mean_fr": float(foot_speed_mean[0]),
        "foot_speed_mean_fl": float(foot_speed_mean[1]),
        "foot_speed_mean_rr": float(foot_speed_mean[2]),
        "foot_speed_mean_rl": float(foot_speed_mean[3]),
        "nonfinite_flag": not finite,
        "reset_reconstruction_error": float(trajectory.get("reset_reconstruction_error", float("nan"))),
        "simulator_mismatch": dict(trajectory.get("simulator_mismatch") or {}),
    }
