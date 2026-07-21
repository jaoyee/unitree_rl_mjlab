from __future__ import annotations

import argparse
import unittest

import torch

from scripts.reinforcement_learning.rwm_trace.audit_trace_replay_ood import (
    ACTION_NAMES,
    _audit_field,
    _command_mode_counts,
    _dataset_observation_chunks,
    _dataset_next_observation_chunks,
    _flat_chunks,
    _reference_stats,
)


class TraceOodAuditTest(unittest.TestCase):
    def test_command_modes_are_classified_from_torch_chunks(self) -> None:
        observations = torch.zeros(8, 48)
        observations[:, 33:36] = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2],
                [0.2, 0.1, 0.0],
                [0.2, 0.0, 0.2],
                [0.0, 0.1, 0.2],
                [0.2, 0.1, 0.2],
            ]
        )
        counts = _command_mode_counts(iter((observations,)))
        self.assertTrue(all(value == 1 for value in counts.values()))

    def test_45d_cached_observation_is_rebuilt_as_48d(self) -> None:
        rows = 4
        data = {
            "observations": torch.full((rows, 45), 99.0),
            "states": torch.zeros(rows, 45),
            "commands": torch.ones(rows, 3),
            "prev_actions": torch.full((rows, 12), 2.0),
        }
        rebuilt = torch.cat(list(_dataset_observation_chunks(data, 2)), dim=0)
        self.assertEqual(tuple(rebuilt.shape), (rows, 48))
        self.assertTrue(torch.all(rebuilt[:, 33:36] == 1.0))
        self.assertTrue(torch.all(rebuilt[:, 36:] == 2.0))

    def test_next_observation_uses_current_action(self) -> None:
        rows = 6
        next_states = torch.zeros(rows, 45)
        commands = torch.ones(rows, 3)
        actions = torch.arange(rows * 12, dtype=torch.float32).reshape(rows, 12)
        data = {
            "next_states": next_states,
            "commands": commands,
            "actions": actions,
        }
        rebuilt = torch.cat(list(_dataset_next_observation_chunks(data, 2)), dim=0)
        self.assertTrue(torch.equal(rebuilt[:, 36:], actions))

    def test_single_action_dimension_cannot_hide_in_aggregate(self) -> None:
        generator = torch.Generator().manual_seed(3)
        reference = torch.randn(400, 12, generator=generator)
        shifted = reference.clone()
        shifted[:, 0] += 20.0
        stats = _reference_stats(
            lambda: _flat_chunks(reference, 12, 64),
            12,
            200,
            1,
        )
        args = argparse.Namespace(
            max_mean_abs_z=5.0,
            max_outside_fraction=0.5,
            max_feature_mean_abs_z=10.0,
            max_feature_outside_fraction=0.9,
        )
        report = _audit_field(
            "action",
            ACTION_NAMES,
            stats,
            _flat_chunks(shifted, 12, 64),
            200,
            2,
            args,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["worst_feature_by_mean_abs_z"]["name"], "action_0")


if __name__ == "__main__":
    unittest.main()
    _dataset_observation_chunks,
