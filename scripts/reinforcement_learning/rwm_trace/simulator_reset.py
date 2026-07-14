"""Approximate reset of MJLab Go2 from an offline 45D RWM state.

The RWM state omits root position and yaw, and actuator force is not a writable
state.  TRACE therefore fixes root x/y/yaw, preserves the default root height,
and reconstructs roll/pitch from projected gravity.  Callers must retain the
reported reconstruction error instead of treating this as an exact reset.
"""

from __future__ import annotations

from typing import Any

import torch


def roll_pitch_from_projected_gravity(projected_gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = projected_gravity / torch.clamp(torch.linalg.norm(projected_gravity, dim=-1, keepdim=True), min=1.0e-8)
    pitch = torch.asin(torch.clamp(gravity[..., 0], -1.0, 1.0))
    roll = torch.atan2(-gravity[..., 1], -gravity[..., 2])
    return roll, pitch


def quaternion_wxyz_from_roll_pitch(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    # yaw is deliberately fixed to zero because it is absent from the RWM state.
    return torch.stack([cr * cp, sr * cp, cr * sp, -sr * sp], dim=-1)


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    first_cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (
        scalar * first_cross + torch.cross(xyz, first_cross, dim=-1)
    )


def reset_go2_from_rwm_state(
    env: Any,
    states: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    robot_name: str = "robot",
) -> dict[str, Any]:
    """Write recoverable RWM state fields into selected MJLab environments."""

    if states.ndim != 2 or states.shape[-1] != 45:
        raise ValueError(f"Expected [batch, 45] RWM states, got {tuple(states.shape)}.")
    if len(states) != len(env_ids):
        raise ValueError("states and env_ids must have the same batch length.")
    robot = env.scene[robot_name]
    states = states.to(device=env.device, dtype=torch.float32)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    roll, pitch = roll_pitch_from_projected_gravity(states[:, 6:9])
    quaternion = quaternion_wxyz_from_roll_pitch(roll, pitch)

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :2] += env.scene.env_origins[env_ids, :2]
    root_state[:, 3:7] = quaternion
    root_state[:, 7:10] = _quat_apply_wxyz(quaternion, states[:, 0:3])
    root_state[:, 10:13] = _quat_apply_wxyz(quaternion, states[:, 3:6])
    joint_position = robot.data.default_joint_pos[env_ids] + states[:, 9:21]
    joint_velocity = states[:, 21:33]

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
    if hasattr(env, "episode_length_buf"):
        env.episode_length_buf[env_ids] = 0
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "_action"):
        env.action_manager._action[env_ids] = 0.0
    if hasattr(env, "sim"):
        env.sim.forward()
    if hasattr(env, "scene") and hasattr(env.scene, "update") and hasattr(env, "step_dt"):
        env.scene.update(env.step_dt)

    return {
        "exact_reset": False,
        "fixed_fields": ["root_x", "root_y", "root_z", "root_yaw"],
        "unwritable_fields": ["actuator_force"],
        "recoverable_state_slice": [0, 33],
    }
