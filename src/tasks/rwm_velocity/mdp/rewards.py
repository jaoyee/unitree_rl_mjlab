"""Tensor-only Go2 velocity rewards for imagination rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .extractors import split_go2_state


@dataclass
class Go2RWMRewardWeights:
    track_linear_velocity: float = 5.0
    track_angular_velocity: float = 3.0
    body_orientation_l2: float = -0.75
    body_ang_vel: float = -0.03
    dof_torques_l2: float = -2.5e-5
    dof_acc_l2: float = -2.5e-7
    action_rate_l2: float = -0.05
    foot_gait: float = 0.4
    stand_still: float = -1.0
    uncertainty: float = -1.0
    command_response: float = 2.0
    yaw_command_response: float = 1.0
    wrong_direction: float = -2.0
    response_shortfall: float = -6.0
    response_floor: float = 0.30
    active_command_bias: float = -0.20
    action_saturation: float = 0.0
    action_saturation_threshold: float = 0.9


@dataclass
class Go2RWMRewardState:
    last_joint_vel: torch.Tensor
    last_action: torch.Tensor
    step_dt: float
    gait_period: float = 0.6
    gait_offsets: torch.Tensor | None = None
    reward_version: str = "v1"
    command_active_threshold: float = 0.02
    motion_gate_low: float = 0.05
    motion_gate_high: float = 0.30
    weights: Go2RWMRewardWeights = field(default_factory=Go2RWMRewardWeights)

    @classmethod
    def create(
        cls,
        num_envs: int,
        action_dim: int,
        device: torch.device | str,
        step_dt: float,
        reward_version: str = "v1",
    ) -> "Go2RWMRewardState":
        return cls(
            last_joint_vel=torch.zeros(num_envs, 12, device=device),
            last_action=torch.zeros(num_envs, action_dim, device=device),
            step_dt=step_dt,
            gait_offsets=torch.tensor([0.0, 0.5, 0.5, 0.0], device=device),
            reward_version=reward_version,
        )


def compute_go2_imagination_reward(
    state: torch.Tensor,
    action: torch.Tensor,
    command: torch.Tensor,
    foot_contact: torch.Tensor,
    episode_length: torch.Tensor,
    reward_state: Go2RWMRewardState,
    epistemic_uncertainty: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts = split_go2_state(state)
    base_lin_vel = parts["base_lin_vel"]
    base_ang_vel = parts["base_ang_vel"]
    projected_gravity = parts["projected_gravity"]
    joint_pos = parts["joint_pos"]
    joint_vel = parts["joint_vel"]
    actuator_force = parts["actuator_force"]

    lin_error = torch.sum(torch.square(command[:, :2] - base_lin_vel[:, :2]), dim=-1)
    lin_error = lin_error + 2.0 * torch.square(base_lin_vel[:, 2])
    track_linear_velocity = torch.exp(-lin_error / 0.25)

    ang_error = torch.square(command[:, 2] - base_ang_vel[:, 2])
    ang_error = ang_error + 0.05 * torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
    track_angular_velocity = torch.exp(-ang_error / 0.5)

    body_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    body_ang_vel = torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
    dof_torques_l2 = torch.sum(torch.square(actuator_force), dim=-1)
    joint_acc = (joint_vel - reward_state.last_joint_vel) / reward_state.step_dt
    dof_acc_l2 = torch.sum(torch.square(joint_acc), dim=-1)
    action_rate_l2 = torch.sum(torch.square(action - reward_state.last_action), dim=-1)
    action_saturation = torch.mean(
        torch.square(torch.relu(torch.abs(action) - float(reward_state.weights.action_saturation_threshold))),
        dim=-1,
    )

    linear_command_norm = torch.linalg.norm(command[:, :2], dim=-1)
    yaw_command_abs = torch.abs(command[:, 2])
    linear_active = linear_command_norm > reward_state.command_active_threshold
    yaw_active = yaw_command_abs > reward_state.command_active_threshold
    hierarchical_active = linear_active | yaw_active
    legacy_active = (linear_command_norm + yaw_command_abs) > 0.1
    active_bool = hierarchical_active if reward_state.reward_version == "v2_hierarchical" else legacy_active
    active = active_bool.float()
    phase = ((episode_length.float() * reward_state.step_dt) / reward_state.gait_period).unsqueeze(-1)
    offsets = reward_state.gait_offsets
    assert offsets is not None
    stance = ((phase + offsets.view(1, -1)) % 1.0) < 0.56
    foot_gait = (stance == foot_contact.bool()).float().mean(dim=-1) * active
    stand_still = torch.sum(torch.square(joint_pos), dim=-1) * (1.0 - active)

    uncertainty = epistemic_uncertainty.reshape(-1)

    linear_static_track = torch.exp(-torch.sum(torch.square(command[:, :2]), dim=-1) / 0.25)
    yaw_static_track = torch.exp(-torch.square(command[:, 2]) / 0.5)
    linear_tracking_advantage = (track_linear_velocity - linear_static_track) * linear_active.float()
    yaw_tracking_advantage = (track_angular_velocity - yaw_static_track) * yaw_active.float()

    linear_projection = torch.sum(base_lin_vel[:, :2] * command[:, :2], dim=-1)
    linear_response = linear_projection / torch.square(linear_command_norm).clamp_min(1e-6)
    yaw_response = (base_ang_vel[:, 2] * command[:, 2]) / torch.square(yaw_command_abs).clamp_min(1e-6)
    inactive_response = torch.full_like(linear_response, float("inf"))
    component_response = torch.minimum(
        torch.where(linear_active, linear_response, inactive_response),
        torch.where(yaw_active, yaw_response, inactive_response),
    )
    component_response = torch.where(hierarchical_active, component_response, torch.ones_like(component_response))
    linear_response_quality = torch.clamp(linear_response, min=0.0, max=1.0) * linear_active.float()
    yaw_response_quality = torch.clamp(yaw_response, min=0.0, max=1.0) * yaw_active.float()
    response_floor = float(reward_state.weights.response_floor)
    linear_response_shortfall = (
        torch.relu(torch.full_like(linear_response, response_floor) - linear_response)
        * linear_active.float()
    )
    yaw_response_shortfall = (
        torch.relu(torch.full_like(yaw_response, response_floor) - yaw_response)
        * yaw_active.float()
    )
    response_shortfall = linear_response_shortfall + yaw_response_shortfall
    wrong_direction = (
        torch.clamp(-linear_response, min=0.0, max=1.0) * linear_active.float()
        + torch.clamp(-yaw_response, min=0.0, max=1.0) * yaw_active.float()
    )
    gate_denominator = max(float(reward_state.motion_gate_high - reward_state.motion_gate_low), 1e-6)
    gate_x = torch.clamp((component_response - float(reward_state.motion_gate_low)) / gate_denominator, 0.0, 1.0)
    motion_gate = (gate_x * gate_x * (3.0 - 2.0 * gate_x)) * active

    terms = {
        "track_linear_velocity": track_linear_velocity,
        "track_angular_velocity": track_angular_velocity,
        "body_orientation_l2": body_orientation_l2,
        "body_ang_vel": body_ang_vel,
        "dof_torques_l2": dof_torques_l2,
        "dof_acc_l2": dof_acc_l2,
        "action_rate_l2": action_rate_l2,
        "action_saturation": action_saturation,
        "foot_gait": foot_gait,
        "stand_still": stand_still,
        "uncertainty": uncertainty,
        "linear_tracking_advantage": linear_tracking_advantage,
        "yaw_tracking_advantage": yaw_tracking_advantage,
        "linear_command_response": linear_response_quality,
        "yaw_command_response": yaw_response_quality,
        "command_response": linear_response_quality + yaw_response_quality,
        "linear_response_shortfall": linear_response_shortfall,
        "yaw_response_shortfall": yaw_response_shortfall,
        "response_shortfall": response_shortfall,
        "wrong_direction": wrong_direction,
        "motion_gate": motion_gate,
    }

    weights = reward_state.weights
    common_quality = (
        weights.track_linear_velocity * track_linear_velocity
        + weights.track_angular_velocity * track_angular_velocity
        + weights.body_orientation_l2 * body_orientation_l2
        + weights.body_ang_vel * body_ang_vel
        + weights.dof_torques_l2 * dof_torques_l2
        + weights.dof_acc_l2 * dof_acc_l2
        + weights.action_rate_l2 * action_rate_l2
        + weights.action_saturation * action_saturation
        + weights.foot_gait * foot_gait
        + weights.stand_still * stand_still
        + weights.uncertainty * uncertainty
    )
    if reward_state.reward_version == "v1":
        reward = common_quality * reward_state.step_dt
    elif reward_state.reward_version == "v2_hierarchical":
        stand_reward = common_quality
        active_axis_count = linear_active.float() + yaw_active.float()
        gated_effort_scale = 0.10 + 0.90 * motion_gate
        active_reward = (
            weights.active_command_bias * active_axis_count
            + weights.command_response * linear_response_quality
            + weights.yaw_command_response * yaw_response_quality
            + weights.response_shortfall * response_shortfall
            + weights.wrong_direction * wrong_direction
            + weights.track_linear_velocity * linear_tracking_advantage
            + weights.track_angular_velocity * yaw_tracking_advantage
            + gated_effort_scale
            * (
                weights.body_orientation_l2 * body_orientation_l2
                + weights.body_ang_vel * body_ang_vel
                + weights.dof_torques_l2 * dof_torques_l2
                + weights.dof_acc_l2 * dof_acc_l2
                + weights.action_rate_l2 * action_rate_l2
                + weights.action_saturation * action_saturation
                + weights.foot_gait * foot_gait
                + weights.uncertainty * uncertainty
            )
        )
        reward = torch.where(active_bool, active_reward, stand_reward) * reward_state.step_dt
    else:
        raise ValueError(f"Unknown Go2 reward version: {reward_state.reward_version!r}")

    reward_state.last_joint_vel = joint_vel.detach()
    reward_state.last_action = action.detach()
    return reward, terms


def bad_orientation_from_state(state: torch.Tensor, limit_angle: float = math.radians(70.0)) -> torch.Tensor:
    projected_gravity = split_go2_state(state)["projected_gravity"]
    return torch.linalg.norm(projected_gravity[:, :2], dim=-1) > math.sin(limit_angle)
