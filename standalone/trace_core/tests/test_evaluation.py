from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core.evaluation import compute_three_metrics


class EvaluationTest(unittest.TestCase):
    def test_fixed_horizon_success_and_survival_steps(self) -> None:
        terminated = np.zeros((5, 3), dtype=bool)
        terminated[2, 0] = True
        terminated[0, 2] = True
        command = np.zeros((5, 3, 3), dtype=np.float64)
        velocity = command.copy()
        velocity[..., 2] = 99.0
        yaw_velocity = np.zeros((5, 3), dtype=np.float64)
        report = compute_three_metrics(
            terminated,
            velocity,
            yaw_velocity,
            command,
        )
        self.assertAlmostEqual(report["success_percent"], 100.0 / 3.0)
        self.assertEqual(report["success_count"], 1)
        self.assertAlmostEqual(report["survival_steps_mean"], 3.0)
        self.assertEqual(report["linear_velocity_error_mps"], 0.0)
        self.assertEqual(report["yaw_velocity_error_radps"], 0.0)

    def test_yaw_error_uses_angular_not_vertical_velocity(self) -> None:
        terminated = np.zeros((2, 1), dtype=bool)
        command = np.zeros((2, 1, 3), dtype=np.float64)
        command[..., 2] = 0.50
        linear_velocity = np.zeros((2, 1, 3), dtype=np.float64)
        linear_velocity[..., 2] = 100.0
        yaw_velocity = np.full((2, 1), 0.25, dtype=np.float64)
        report = compute_three_metrics(
            terminated,
            linear_velocity,
            yaw_velocity,
            command,
        )
        self.assertAlmostEqual(report["yaw_velocity_error_radps"], 0.25)


if __name__ == "__main__":
    unittest.main()
