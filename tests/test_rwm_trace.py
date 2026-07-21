from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.reinforcement_learning.rwm_trace.build_trace_replay import (
    _candidate_windows,
    _n_step_rows,
)
from scripts.reinforcement_learning.rwm_trace.replay import TraceReplaySampler, mix_trace_replay_batch
from scripts.reinforcement_learning.rwm_trace.scorer import (
    GO2_FEATURE_NAMES,
    Go2TraceScorer,
    feature_matrix,
    fit_feature_stats,
    load_scorer_checkpoint,
    save_scorer_checkpoint,
    score_summaries,
)
from scripts.reinforcement_learning.rwm_trace.trajectory import summarize_go2_trajectory
from scripts.reinforcement_learning.rwm_trace.train_go2_trace_scorer import (
    _zero_all_missing_input_weights,
)
from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
    quaternion_wxyz_from_roll_pitch,
    roll_pitch_from_projected_gravity,
)


class TraceScorerTest(unittest.TestCase):
    def test_all_missing_privileged_features_have_zero_input_weights(self) -> None:
        feature_count = len(GO2_FEATURE_NAMES)
        features = np.zeros((4, 2 * feature_count), dtype=np.float32)
        missing_index = GO2_FEATURE_NAMES.index("foot_height_range_rr")
        features[:, feature_count + missing_index] = 1.0
        model = Go2TraceScorer(2 * feature_count, hidden_dim=8)
        names = _zero_all_missing_input_weights(model, features)
        self.assertIn("foot_height_range_rr", names)
        self.assertTrue(torch.all(model.net[0].weight[:, missing_index] == 0.0))
        self.assertTrue(
            torch.all(model.net[0].weight[:, feature_count + missing_index] == 0.0)
        )

    def test_summary_and_checkpoint_round_trip(self) -> None:
        length = 20
        states = torch.zeros(length, 45)
        next_states = states.clone()
        next_states[:, 0] = 0.2
        next_states[:, 8] = -1.0
        next_states[:, 17] = -3.0
        next_states[:, 20] = 0.75
        trajectory = {
            "trajectory_id": "candidate-1",
            "states": states,
            "next_states": next_states,
            "actions": torch.zeros(length, 12),
            "rewards": torch.ones(length),
            "commands": torch.tensor([[0.2, 0.0, 0.0]]).repeat(length, 1),
            "contacts": torch.ones(length, 4),
            "terminations": torch.zeros(length, 1),
        }
        summary = summarize_go2_trajectory(trajectory)
        self.assertAlmostEqual(summary["simulator_return"], 20.0)
        self.assertAlmostEqual(summary["linear_tracking_error_mean"], 0.0)
        self.assertAlmostEqual(summary["rr_calf_relative_position_mean"], 0.75)
        features = feature_matrix([summary, summary], GO2_FEATURE_NAMES)
        stats = fit_feature_stats(features)
        model = Go2TraceScorer(features.shape[-1], hidden_dim=8)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "scorer.pt"
            save_scorer_checkpoint(str(checkpoint), model, stats, hidden_dim=8)
            loaded, loaded_stats, _ = load_scorer_checkpoint(str(checkpoint))
            scores = score_summaries(loaded, [summary], loaded_stats)
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(np.isfinite(scores).all())

    def test_projected_gravity_reconstruction(self) -> None:
        gravity = torch.tensor([[0.0, 0.0, -1.0], [0.2, -0.1, -0.9746794]])
        roll, pitch = roll_pitch_from_projected_gravity(gravity)
        quaternion = quaternion_wxyz_from_roll_pitch(roll, pitch)
        self.assertEqual(tuple(quaternion.shape), (2, 4))
        self.assertTrue(torch.isfinite(quaternion).all())
        self.assertTrue(torch.allclose(torch.linalg.norm(quaternion, dim=-1), torch.ones(2), atol=1.0e-5))


class TraceReplayTest(unittest.TestCase):
    def test_candidate_windows_and_n_step_targets(self) -> None:
        time_steps, num_envs = 8, 2
        dataset = {
            "states": [torch.zeros(num_envs, 45) for _ in range(time_steps)],
            "actions": [torch.zeros(num_envs, 12) for _ in range(time_steps)],
            "next_states": [torch.zeros(num_envs, 45) for _ in range(time_steps)],
            "contacts": [torch.ones(num_envs, 4) for _ in range(time_steps)],
            "terminations": [torch.zeros(num_envs, 1) for _ in range(time_steps)],
            "commands": [torch.zeros(num_envs, 3) for _ in range(time_steps)],
            "rewards": [torch.ones(num_envs) for _ in range(time_steps)],
            "prev_actions": [torch.zeros(num_envs, 12) for _ in range(time_steps)],
            "trace_start_state_ids": torch.tensor([5, 5]),
        }
        candidates = _candidate_windows(dataset, length=4, stride=4)
        self.assertEqual(len(candidates), 4)
        self.assertEqual({candidate["start_state_id"] for candidate in candidates}, {5})
        rows = _n_step_rows(candidates[0], n_step=3, gamma=1.0)
        self.assertEqual(len(rows["reward"]), 1)
        self.assertTrue(torch.equal(torch.stack(rows["reward"]), torch.tensor([3.0])))
        self.assertEqual(tuple(rows["observation"][0].shape), (48,))

    def test_fixed_ratio_replacement(self) -> None:
        artifact = {
            "format_version": "go2_trace_replay_v1",
            "observation": torch.full((10, 48), 7.0),
            "action": torch.full((10, 12), 7.0),
            "reward": torch.full((10,), 7.0),
            "terminated": torch.zeros(10),
            "truncated": torch.zeros(10),
            "next_observation": torch.full((10, 48), 7.0),
        }
        batch = {
            "observation": torch.zeros(8, 48),
            "action": torch.zeros(8, 12),
            "reward": torch.zeros(8),
            "terminated": torch.zeros(8),
            "truncated": torch.zeros(8),
            "next_observation": torch.zeros(8, 48),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pt"
            torch.save(artifact, path)
            sampler = TraceReplaySampler(path, seed=3)
            mixed, count = mix_trace_replay_batch(batch, sampler, 0.25)
        self.assertEqual(count, 2)
        self.assertTrue(torch.all(mixed["reward"][:2] == 7.0))
        self.assertTrue(torch.all(mixed["reward"][2:] == 0.0))


if __name__ == "__main__":
    unittest.main()
