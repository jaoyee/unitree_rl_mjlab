from __future__ import annotations

import numpy as np

from scripts.reinforcement_learning.rwm_flashsac.command_metrics_v4 import (
    summarize_command_segments,
)


def test_unresponsive_policy_does_not_pass_on_episode_survival() -> None:
    steps, envs = 40, 2
    lin = np.zeros((steps, envs, 3), dtype=np.float32)
    yaw = np.zeros((steps, envs), dtype=np.float32)
    commands = np.zeros((steps, envs, 3), dtype=np.float32)
    commands[20:, :, 0] = 0.5
    lin[20:, 0, 0] = 0.5
    actions = np.zeros((steps, envs, 12), dtype=np.float32)
    result = summarize_command_segments(
        base_lin_vel=lin,
        base_yaw_vel=yaw,
        commands=commands,
        terminated=np.zeros((steps, envs), dtype=bool),
        truncated=np.zeros((steps, envs), dtype=bool),
        actions=actions,
        command_sequence=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        command_switch_steps=20,
        step_dt=0.02,
        settle_steps=0,
        sustained_response_steps=2,
    )
    forward = result["per_command"][1]
    assert forward["pass_rate"] == 0.5
    assert forward["no_response_rate"] == 0.5
    assert forward["response_success_rate"] == 0.5
    assert result["all_commands_pass_rate"] == 0.5


def test_termination_and_action_saturation_are_reported_per_segment() -> None:
    steps, envs = 20, 2
    lin = np.zeros((steps, envs, 3), dtype=np.float32)
    yaw = np.full((steps, envs), 0.4, dtype=np.float32)
    commands = np.zeros((steps, envs, 3), dtype=np.float32)
    commands[:, :, 2] = 0.4
    terminated = np.zeros((steps, envs), dtype=bool)
    terminated[-1, 1] = True
    actions = np.zeros((steps, envs, 12), dtype=np.float32)
    actions[:, 0, 0] = 1.0
    result = summarize_command_segments(
        base_lin_vel=lin,
        base_yaw_vel=yaw,
        commands=commands,
        terminated=terminated,
        truncated=np.zeros((steps, envs), dtype=bool),
        actions=actions,
        command_sequence=((0.0, 0.0, 0.4),),
        command_switch_steps=20,
        step_dt=0.02,
        settle_steps=0,
        sustained_response_steps=2,
    )
    segment = result["per_command"][0]
    assert segment["termination_rate"] == 0.5
    assert segment["pass_rate"] == 0.5
    assert segment["action_saturation_transition_rate"] == 0.5


def test_effective_swing_requires_contact_cycle_height_and_speed() -> None:
    steps, envs = 20, 1
    lin = np.zeros((steps, envs, 3), dtype=np.float32)
    lin[:, :, 0] = 0.3
    commands = np.zeros((steps, envs, 3), dtype=np.float32)
    commands[:, :, 0] = 0.3
    contacts = np.ones((steps, envs, 4), dtype=bool)
    contacts[5:10, :, 0] = False
    heights = np.zeros((steps, envs, 4), dtype=np.float32)
    heights[5:10, :, 0] = 0.02
    speeds = np.zeros((steps, envs, 4), dtype=np.float32)
    speeds[5:10, :, 0] = 0.1
    result = summarize_command_segments(
        base_lin_vel=lin,
        base_yaw_vel=np.zeros((steps, envs), dtype=np.float32),
        commands=commands,
        terminated=np.zeros((steps, envs), dtype=bool),
        truncated=np.zeros((steps, envs), dtype=bool),
        actions=np.zeros((steps, envs, 12), dtype=np.float32),
        command_sequence=((0.3, 0.0, 0.0),),
        command_switch_steps=steps,
        step_dt=0.02,
        foot_contacts=contacts,
        foot_heights=heights,
        foot_speeds=speeds,
        settle_steps=0,
        sustained_response_steps=2,
    )
    assert result["mean_effective_swing_foot_fraction_nonstand"] == 0.25
    assert result["mean_no_effective_swing_rate_nonstand"] == 0.0
