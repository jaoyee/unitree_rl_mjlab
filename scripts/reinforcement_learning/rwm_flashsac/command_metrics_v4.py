"""Command-segment metrics that reject stable-but-unresponsive locomotion policies."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _first_sustained_true(mask: np.ndarray, required_steps: int) -> np.ndarray:
    """Return the first sustained-true index for every environment, or -1."""

    if mask.ndim != 2:
        raise ValueError(f"Expected [steps, envs] mask, got shape={mask.shape}.")
    required_steps = max(1, int(required_steps))
    result = np.full(mask.shape[1], -1, dtype=np.int64)
    if mask.shape[0] < required_steps:
        return result
    running = np.zeros(mask.shape[1], dtype=np.int64)
    for step in range(mask.shape[0]):
        running = np.where(mask[step], running + 1, 0)
        newly_sustained = (result < 0) & (running >= required_steps)
        result[newly_sustained] = step - required_steps + 1
    return result


def summarize_command_segments(
    *,
    base_lin_vel: np.ndarray,
    base_yaw_vel: np.ndarray,
    commands: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    actions: np.ndarray,
    command_sequence: Sequence[Sequence[float]],
    command_switch_steps: int,
    step_dt: float,
    settle_steps: int = 50,
    sustained_response_steps: int = 10,
    xy_error_threshold: float = 0.15,
    yaw_error_threshold: float = 0.15,
    stand_xy_speed_threshold: float = 0.10,
    stand_yaw_speed_threshold: float = 0.10,
    minimum_response_gain: float = 0.50,
    maximum_response_gain: float = 1.50,
    no_response_gain: float = 0.20,
    action_saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Build per-command metrics and hard pass rates.

    All time-series arrays are expected to be shaped ``[steps, envs, ...]``.
    Displacement is integrated in the body frame. This remains meaningful when
    an environment auto-resets after a fall and avoids pose-wrap bookkeeping.
    """

    base_lin_vel = np.asarray(base_lin_vel, dtype=np.float64)
    base_yaw_vel = np.asarray(base_yaw_vel, dtype=np.float64)
    commands = np.asarray(commands, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    actions = np.asarray(actions, dtype=np.float64)
    if base_lin_vel.ndim != 3 or base_lin_vel.shape[-1] < 2:
        raise ValueError(f"base_lin_vel must be [steps, envs, >=2], got {base_lin_vel.shape}.")
    steps, num_envs = base_lin_vel.shape[:2]
    expected_shapes = {
        "base_yaw_vel": (steps, num_envs),
        "commands": (steps, num_envs, 3),
        "terminated": (steps, num_envs),
        "truncated": (steps, num_envs),
    }
    actual_shapes = {
        "base_yaw_vel": base_yaw_vel.shape,
        "commands": commands.shape,
        "terminated": terminated.shape,
        "truncated": truncated.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual_shapes[name]}.")
    if actions.ndim != 3 or actions.shape[:2] != (steps, num_envs):
        raise ValueError(f"actions must be [steps, envs, action_dim], got {actions.shape}.")
    if not command_sequence:
        return {}
    switch_steps = max(1, int(command_switch_steps))
    segment_count = min(len(command_sequence), (steps + switch_steps - 1) // switch_steps)
    all_segment_passes = np.ones((num_envs, segment_count), dtype=bool)
    per_command: list[dict[str, Any]] = []

    for segment_index in range(segment_count):
        start = segment_index * switch_steps
        stop = min(steps, start + switch_steps)
        active_start = min(stop - 1, start + max(0, int(settle_steps)))
        active = slice(active_start, stop)
        duration_steps = stop - active_start
        duration_seconds = duration_steps * float(step_dt)
        command = np.asarray(command_sequence[segment_index], dtype=np.float64)
        lin_command = command[:2]
        lin_norm = float(np.linalg.norm(lin_command))
        yaw_command = float(command[2])
        lin_active = lin_norm > 1.0e-6
        yaw_active = abs(yaw_command) > 1.0e-6

        velocity_xy = base_lin_vel[active, :, :2]
        velocity_yaw = base_yaw_vel[active]
        xy_error = np.linalg.norm(velocity_xy - lin_command.reshape(1, 1, 2), axis=-1)
        yaw_error = np.abs(velocity_yaw - yaw_command)
        mean_xy_error_per_env = xy_error.mean(axis=0)
        mean_yaw_error_per_env = yaw_error.mean(axis=0)
        mean_speed_per_env = np.linalg.norm(velocity_xy, axis=-1).mean(axis=0)
        mean_abs_yaw_per_env = np.abs(velocity_yaw).mean(axis=0)

        if lin_active:
            lin_direction = lin_command / lin_norm
            projected_velocity = np.sum(velocity_xy * lin_direction.reshape(1, 1, 2), axis=-1)
            linear_gain_per_env = projected_velocity.mean(axis=0) / lin_norm
            displacement_per_env = projected_velocity.sum(axis=0) * float(step_dt)
            expected_displacement = lin_norm * duration_seconds
            linear_pass = (
                (mean_xy_error_per_env <= xy_error_threshold)
                & (linear_gain_per_env >= minimum_response_gain)
                & (linear_gain_per_env <= maximum_response_gain)
            )
        else:
            projected_velocity = np.zeros((duration_steps, num_envs), dtype=np.float64)
            linear_gain_per_env = np.ones(num_envs, dtype=np.float64)
            displacement_per_env = np.linalg.norm(velocity_xy, axis=-1).sum(axis=0) * float(step_dt)
            expected_displacement = 0.0
            linear_pass = mean_speed_per_env <= stand_xy_speed_threshold

        if yaw_active:
            yaw_gain_per_env = velocity_yaw.mean(axis=0) / yaw_command
            yaw_progress_per_env = velocity_yaw.sum(axis=0) * float(step_dt)
            expected_yaw_progress = yaw_command * duration_seconds
            yaw_pass = (
                (mean_yaw_error_per_env <= yaw_error_threshold)
                & (yaw_gain_per_env >= minimum_response_gain)
                & (yaw_gain_per_env <= maximum_response_gain)
            )
        else:
            yaw_gain_per_env = np.ones(num_envs, dtype=np.float64)
            yaw_progress_per_env = np.abs(velocity_yaw).sum(axis=0) * float(step_dt)
            expected_yaw_progress = 0.0
            yaw_pass = mean_abs_yaw_per_env <= stand_yaw_speed_threshold

        segment_terminated = terminated[start:stop].any(axis=0)
        segment_truncated = truncated[start:stop].any(axis=0)
        segment_pass = linear_pass & yaw_pass & ~segment_terminated
        all_segment_passes[:, segment_index] = segment_pass

        instantaneous_ok = (xy_error <= xy_error_threshold) & (yaw_error <= yaw_error_threshold)
        if lin_active:
            instantaneous_ok &= projected_velocity >= minimum_response_gain * lin_norm
        if yaw_active:
            instantaneous_ok &= (velocity_yaw / yaw_command) >= minimum_response_gain
        latency_indices = _first_sustained_true(instantaneous_ok, sustained_response_steps)
        latency_seconds = np.where(
            latency_indices >= 0,
            (latency_indices + max(0, int(settle_steps))) * float(step_dt),
            np.nan,
        )

        no_response = np.zeros(num_envs, dtype=bool)
        if lin_active:
            no_response |= linear_gain_per_env < no_response_gain
        if yaw_active:
            no_response |= yaw_gain_per_env < no_response_gain
        segment_actions = actions[start:stop]
        saturation = (np.abs(segment_actions) > action_saturation_threshold).any(axis=-1)
        if stop - start > 1:
            action_delta = np.abs(np.diff(segment_actions, axis=0)).mean(axis=-1)
            action_delta_mean = float(action_delta.mean())
        else:
            action_delta_mean = 0.0

        valid_latency = latency_seconds[np.isfinite(latency_seconds)]
        per_command.append(
            {
                "index": segment_index,
                "command": command.tolist(),
                "start_step": start,
                "stop_step": stop,
                "settle_steps": max(0, int(settle_steps)),
                "mean_error_vel_xy": float(mean_xy_error_per_env.mean()),
                "mean_error_vel_yaw": float(mean_yaw_error_per_env.mean()),
                "linear_response_gain_mean": float(linear_gain_per_env.mean()),
                "yaw_response_gain_mean": float(yaw_gain_per_env.mean()),
                "body_frame_progress_mean": float(displacement_per_env.mean()),
                "expected_body_frame_progress": float(expected_displacement),
                "yaw_progress_mean": float(yaw_progress_per_env.mean()),
                "expected_yaw_progress": float(expected_yaw_progress),
                "response_latency_mean_s": (
                    float(valid_latency.mean()) if valid_latency.size else None
                ),
                "response_success_rate": float(np.isfinite(latency_seconds).mean()),
                "no_response_rate": float(no_response.mean()),
                "termination_rate": float(segment_terminated.mean()),
                "timeout_rate": float(segment_truncated.mean()),
                "pass_rate": float(segment_pass.mean()),
                "action_saturation_transition_rate": float(saturation.mean()),
                "action_delta_abs_mean": action_delta_mean,
            }
        )

    segment_pass_rates = [entry["pass_rate"] for entry in per_command]
    nonzero_indices = [
        index
        for index, command in enumerate(command_sequence[:segment_count])
        if np.linalg.norm(np.asarray(command, dtype=np.float64)) > 1.0e-6
    ]
    return {
        "per_command": per_command,
        "mean_segment_pass_rate": float(np.mean(segment_pass_rates)),
        "minimum_segment_pass_rate": float(np.min(segment_pass_rates)),
        "all_commands_pass_rate": float(all_segment_passes.all(axis=1).mean()),
        "all_nonzero_commands_pass_rate": (
            float(all_segment_passes[:, nonzero_indices].all(axis=1).mean())
            if nonzero_indices
            else 1.0
        ),
        "mean_no_response_rate": float(
            np.mean([per_command[index]["no_response_rate"] for index in nonzero_indices])
        )
        if nonzero_indices
        else 0.0,
        "thresholds": {
            "xy_error": float(xy_error_threshold),
            "yaw_error": float(yaw_error_threshold),
            "stand_xy_speed": float(stand_xy_speed_threshold),
            "stand_yaw_speed": float(stand_yaw_speed_threshold),
            "minimum_response_gain": float(minimum_response_gain),
            "maximum_response_gain": float(maximum_response_gain),
            "no_response_gain": float(no_response_gain),
            "action_saturation": float(action_saturation_threshold),
            "settle_steps": int(settle_steps),
            "sustained_response_steps": int(sustained_response_steps),
        },
    }
