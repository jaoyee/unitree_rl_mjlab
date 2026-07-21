from __future__ import annotations

import argparse

import numpy as np
import torch

from scripts.reinforcement_learning.rwm_trace.build_trace_replay_v4 import (
    _add_action_safety_summary,
    _apply_replay_safety_costs,
    _n_step_rows,
    _select_indices,
    _validate_controlled_candidates,
)
from scripts.reinforcement_learning.rwm_trace.train_go2_trace_scorer import _component_group_keys


def test_failure_cost_is_ramped_before_terminal() -> None:
    rewards = torch.ones(6)
    actions = torch.zeros(6, 2)
    previous_actions = torch.zeros_like(actions)
    terminated = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32)
    shaped = _apply_replay_safety_costs(
        rewards,
        actions,
        previous_actions,
        terminated,
        terminal_penalty=-10.0,
        override_terminal_reward=True,
        failure_backprop_steps=4,
        failure_backprop_penalty=4.0,
        action_saturation_threshold=0.95,
        action_saturation_penalty_scale=0.0,
        action_delta_penalty_scale=0.0,
    )
    assert torch.allclose(shaped, torch.tensor([1.0, 0.0, -1.0, -2.0, -3.0, -10.0]))


def test_saturation_filter_excludes_unsafe_trajectory() -> None:
    summaries = [
        {"terminal_flag": False, "action_saturation_fraction": 0.1},
        {"terminal_flag": False, "action_saturation_fraction": 0.8},
    ]
    args = argparse.Namespace(
        selection="all",
        max_saturation_fraction=0.2,
        seed=0,
        select_ratio=1.0,
        failure_trajectory_ratio=0.0,
        scorer_checkpoint=None,
    )
    selected, _ = _select_indices(summaries, args)
    assert np.array_equal(selected, np.asarray([0]))


def _selection_args(**overrides: object) -> argparse.Namespace:
    values = {
        "selection": "random",
        "selection_scope": "per_start",
        "max_saturation_fraction": 1.0,
        "seed": 0,
        "select_ratio": 0.25,
        "per_start_target_count": 0,
        "failure_trajectory_ratio": 0.0,
        "failure_transition_ratio": 0.0,
        "scorer_checkpoint": None,
        "command_motion_gate": True,
        "min_linear_realization_ratio": 0.2,
        "min_yaw_realization_ratio": 0.2,
        "max_command_direction_violation_rate": 0.5,
        "command_mode_weights": "pure_x:0.5,x_yaw:0.5",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _motion_summary(group: str, mode: str, *, realization: float, direction: float = 0.0) -> dict:
    return {
        "comparison_group_key": group,
        "command_mode": mode,
        "command_linear_active": True,
        "command_yaw_active": mode == "x_yaw",
        "linear_velocity_realization_ratio_mean": realization,
        "yaw_velocity_realization_ratio_mean": realization,
        "command_direction_violation_rate": direction,
        "command_projected_displacement": realization,
        "yaw_net_change": realization if mode == "x_yaw" else 0.0,
        "yaw_expected_change": 1.0 if mode == "x_yaw" else 0.0,
        "terminal_flag": False,
        "action_saturation_fraction": 0.0,
    }


def test_per_start_selection_never_uses_motion_gate_fallback() -> None:
    summaries = [
        _motion_summary("x0", "pure_x", realization=value)
        for value in (0.1, 0.3, 0.4, 0.5)
    ] + [
        _motion_summary("xy0", "x_yaw", realization=value)
        for value in (0.05, 0.08, 0.1, 0.15)
    ]
    with np.testing.assert_raises_regex(ValueError, "more rollout branches"):
        _select_indices(summaries, _selection_args())


def test_per_start_selection_replaces_group_with_only_severe_direction_errors() -> None:
    summaries = [
        _motion_summary("x0", "pure_x", realization=0.4),
        _motion_summary("x0", "pure_x", realization=0.4),
        _motion_summary("bad", "x_yaw", realization=0.4, direction=0.95),
        _motion_summary("bad", "x_yaw", realization=0.4, direction=0.95),
    ]
    args = _selection_args(select_ratio=0.5)
    with np.testing.assert_raises_regex(ValueError, "more rollout branches"):
        _select_indices(summaries, args)


def test_per_start_selection_keeps_exact_top_quartile_without_fallback() -> None:
    summaries = []
    for group in ("x0", "x1"):
        summaries.extend(
            _motion_summary(group, "pure_x", realization=value)
            for value in (0.1, 0.3, 0.4, 0.5)
        )
    selected, _ = _select_indices(summaries, _selection_args(command_mode_weights=None))
    assert len(selected) == 2
    assert all(summaries[int(index)]["linear_velocity_realization_ratio_mean"] >= 0.2 for index in selected)
    assert _selection_args().command_motion_gate


def test_appended_branches_do_not_increase_fixed_per_start_target() -> None:
    summaries = [
        _motion_summary("x0", "pure_x", realization=value)
        for value in (0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    ]
    selected, _ = _select_indices(
        summaries,
        _selection_args(per_start_target_count=2, command_mode_weights=None),
    )
    assert len(selected) == 2


def test_compound_command_rejects_missing_yaw_response() -> None:
    summary = _motion_summary("xy0", "x_yaw", realization=0.4)
    summary["yaw_net_change"] = 0.0
    with np.testing.assert_raises_regex(ValueError, "rejected every candidate"):
        _select_indices(
            [summary, summary.copy(), summary.copy(), summary.copy()],
            _selection_args(per_start_target_count=1, command_mode_weights=None),
        )


def test_terminal_reward_is_preserved_without_legacy_override() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])
    actions = torch.zeros(3, 2)
    shaped = _apply_replay_safety_costs(
        rewards,
        actions,
        actions,
        torch.tensor([0, 0, 1]),
        terminal_penalty=-10.0,
        override_terminal_reward=False,
        failure_backprop_steps=0,
        failure_backprop_penalty=0.0,
        action_saturation_threshold=0.95,
        action_saturation_penalty_scale=0.0,
        action_delta_penalty_scale=0.0,
    )
    assert torch.equal(shaped, rewards)


def test_per_start_mode_allocation_never_silently_shrinks() -> None:
    summaries = []
    for group in range(2):
        summaries.extend(_motion_summary(f"x{group}", "pure_x", realization=0.4) for _ in range(4))
    for group in range(2):
        summaries.extend(_motion_summary(f"xy{group}", "x_yaw", realization=0.4) for _ in range(4))
    selected, _ = _select_indices(summaries, _selection_args())
    assert len(selected) == 4
    assert sum(summaries[int(index)]["command_mode"] == "pure_x" for index in selected) == 2
    assert sum(summaries[int(index)]["command_mode"] == "x_yaw" for index in selected) == 2


def test_action_safety_summary_uses_any_saturated_dimension() -> None:
    trajectory = {
        "actions": torch.tensor([[1.0, 0.0], [0.1, 0.2]]),
        "prev_actions": torch.zeros(2, 2),
    }
    summary: dict[str, float] = {}
    _add_action_safety_summary(summary, trajectory, 0.95)
    assert summary["action_saturation_fraction"] == 0.5


def test_terminal_n_step_flush_keeps_direct_failure_action() -> None:
    length = 5
    trajectory = {
        "states": torch.zeros(length, 45),
        "actions": torch.arange(length, dtype=torch.float32).view(-1, 1).repeat(1, 12),
        "next_states": torch.zeros(length, 45),
        "commands": torch.zeros(length, 3),
        "prev_actions": torch.zeros(length, 12),
        "rewards": torch.ones(length),
        "terminations": torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32),
    }
    rows = _n_step_rows(trajectory, n_step=3, gamma=1.0)
    assert len(rows["action"]) == length
    assert float(rows["action"][-1][0]) == 4.0
    assert float(rows["reward"][-1]) == 1.0
    assert float(rows["terminated"][-1]) == 1.0


def test_nonterminal_n_step_keeps_last_complete_window() -> None:
    length = 5
    trajectory = {
        "states": torch.zeros(length, 45),
        "actions": torch.zeros(length, 12),
        "next_states": torch.zeros(length, 45),
        "commands": torch.zeros(length, 3),
        "prev_actions": torch.zeros(length, 12),
        "rewards": torch.ones(length),
        "terminations": torch.zeros(length),
    }
    rows = _n_step_rows(trajectory, n_step=3, gamma=1.0)
    assert len(rows["action"]) == 3
    assert torch.allclose(torch.stack(rows["reward"]), torch.full((3,), 3.0))


def test_controlled_candidate_gate_rejects_legacy_artifact() -> None:
    with np.testing.assert_raises_regex(ValueError, "controlled TRACE gates"):
        _validate_controlled_candidates({"metadata": {}}, allow_legacy=False)


def _minimal_controlled_candidate(reset_mode: str) -> dict:
    exact = reset_mode == "exact_snapshot"
    return {
        "metadata": {
            "trace_candidates": {
                "enabled": True,
                "controlled_branch_domain": True,
                "command_from_source_transition": True,
                "stop_on_done": True,
                "reset_mode": reset_mode,
                "reset_is_exact": exact,
                "one_step_identity": {"passed": True},
                "source_transition_identity": {
                    "available": exact,
                    "passed": True if exact else None,
                },
            }
        },
        "states": [torch.zeros(1, 45)],
        "commands": [torch.zeros(1, 3)],
        "prev_actions": [torch.zeros(1, 12)],
        "trace_valid_masks": [torch.ones(1, dtype=torch.bool)],
        "trace_start_state_ids": torch.tensor([0]),
    }


def test_exact_candidate_gate_requires_dataset_next_state_identity() -> None:
    candidate = _minimal_controlled_candidate("exact_snapshot")
    candidate["metadata"]["trace_candidates"]["source_transition_identity"]["passed"] = False
    with np.testing.assert_raises_regex(ValueError, "source next_state"):
        _validate_controlled_candidates(candidate, allow_legacy=False)


def test_canonical_real_gate_must_not_claim_exact_identity() -> None:
    candidate = _minimal_controlled_candidate("canonical_real_projection")
    candidate["metadata"]["trace_candidates"]["reset_is_exact"] = True
    with np.testing.assert_raises_regex(ValueError, "incorrectly claims an exact reset"):
        _validate_controlled_candidates(candidate, allow_legacy=False)


def test_component_split_groups_connected_cross_start_pairs() -> None:
    summaries = [
        {"start_state_id": 1}, {"start_state_id": 2},
        {"start_state_id": 2}, {"start_state_id": 3},
        {"start_state_id": 8}, {"start_state_id": 8},
    ]
    keys = _component_group_keys(summaries, pair_count=3)
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]
