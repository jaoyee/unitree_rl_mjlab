from __future__ import annotations

import torch

from scripts.reinforcement_learning.rwm_dataset.audit_go2_sequence_quality_v4 import audit


def test_sequence_and_termination_support() -> None:
    steps = 12
    dataset = {
        "states": torch.zeros(steps, 1, 45),
        "actions": torch.zeros(steps, 1, 12),
        "commands": torch.zeros(steps, 1, 3),
        "terminations": torch.zeros(steps, 1, 1),
        "episode_ids": torch.tensor([0] * 5 + [1] * 7).reshape(steps, 1),
        "timesteps": torch.tensor(list(range(5)) + list(range(7))).reshape(steps, 1),
    }
    dataset["terminations"][4, 0, 0] = 1.0
    dataset["commands"][5:, 0, 0] = 0.5
    report = audit(dataset, [4, 6], 1.0e-3, 0.95)
    assert report["termination_positive_transitions"] == 1
    assert report["terminal_contiguous_runs"] == 1
    assert report["contiguous_runs"]["count"] == 2
    assert report["sequence_support"]["6"]["eligible_runs"] == 1
    assert report["sequence_support"]["6"]["valid_starts"] == 2
    assert report["command_mode_counts"]["stand"] == 5
    assert report["command_mode_counts"]["pure_x"] == 7
