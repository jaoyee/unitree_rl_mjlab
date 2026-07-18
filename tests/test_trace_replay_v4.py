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
