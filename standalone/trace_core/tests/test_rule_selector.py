from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core.annotate_eligibility import _motion_rejection
from trace_core.rule_score import rule_components


def active_row(**overrides):
    row = {
        "command_linear_active": True,
        "command_yaw_active": False,
        "command_vx_mean": 0.3,
        "command_vy_mean": 0.0,
        "command_yaw_mean": 0.0,
        "command_linear_speed_mean": 0.3,
        "minimum_active_axis_response_ratio": 0.8,
        "minimum_active_axis_transport_ratio": 0.8,
        "linear_tracking_error_mean": 0.05,
        "linear_tracking_error_steady": 0.04,
        "command_direction_correct_fraction": 0.95,
        "command_direction_violation_rate": 0.05,
        "survival_fraction": 1.0,
        "tilt_mean": 0.08,
        "roll_pitch_rate_rms": 0.2,
        "action_delta_norm_mean": 0.3,
        "action_saturation_fraction": 0.0,
        "trace_selection_eligible": True,
    }
    row.update(overrides)
    return row


class RuleSelectorTest(unittest.TestCase):
    def test_motion_dominates_stationary_stability(self) -> None:
        moving = rule_components(active_row())
        stationary = rule_components(
            active_row(
                minimum_active_axis_response_ratio=0.0,
                minimum_active_axis_transport_ratio=0.0,
                linear_tracking_error_mean=0.3,
                linear_tracking_error_steady=0.3,
                tilt_mean=0.0,
                roll_pitch_rate_rms=0.0,
                action_delta_norm_mean=0.0,
                trace_selection_eligible=False,
            )
        )
        self.assertGreater(moving["rule_score"], stationary["rule_score"])
        self.assertEqual(stationary["primary_quality"], 0.0)

    def test_weakest_combined_component_controls_score(self) -> None:
        good = rule_components(
            active_row(
                command_yaw_active=True,
                command_yaw_mean=0.3,
                yaw_tracking_error_mean=0.04,
                yaw_tracking_error_steady=0.04,
            )
        )
        partial = rule_components(
            active_row(
                command_yaw_active=True,
                command_yaw_mean=0.3,
                minimum_active_axis_response_ratio=0.05,
                yaw_tracking_error_mean=0.04,
                yaw_tracking_error_steady=0.04,
            )
        )
        self.assertGreater(good["rule_score"], partial["rule_score"])

    def test_component_gate_is_fail_closed(self) -> None:
        validity = {
            "minimum_component_realization_ratio": 0.1,
            "maximum_direction_violation_rate": 0.5,
        }
        reason = _motion_rejection(
            active_row(minimum_active_axis_response_ratio=0.05),
            validity,
        )
        self.assertEqual(reason, "insufficient_component_response")

    def test_stand_is_scored_separately(self) -> None:
        stable = rule_components(
            {
                "command_linear_active": False,
                "command_yaw_active": False,
                "linear_tracking_error_mean": 0.002,
                "yaw_tracking_error_mean": 0.002,
                "survival_fraction": 1.0,
                "tilt_mean": 0.01,
                "roll_pitch_rate_rms": 0.02,
                "action_delta_norm_mean": 0.02,
                "action_saturation_fraction": 0.0,
                "trace_selection_eligible": True,
            }
        )
        jitter = rule_components(
            {
                "command_linear_active": False,
                "command_yaw_active": False,
                "linear_tracking_error_mean": 0.08,
                "yaw_tracking_error_mean": 0.08,
                "survival_fraction": 1.0,
                "tilt_mean": 0.2,
                "roll_pitch_rate_rms": 1.0,
                "action_delta_norm_mean": 1.0,
                "action_saturation_fraction": 0.1,
                "trace_selection_eligible": True,
            }
        )
        self.assertEqual(stable["profile"], "stand")
        self.assertGreater(stable["rule_score"], jitter["rule_score"])


if __name__ == "__main__":
    unittest.main()
