from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core.export_selection_manifest import main as export_main
from trace_core.rule_score import main as score_main


class RuleArtifactTest(unittest.TestCase):
    def test_rule_scores_export_to_portable_selection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summaries = root / "summaries.jsonl"
            scores = root / "scores.jsonl"
            score_manifest = root / "score_manifest.json"
            scored_summaries = root / "scored_summaries.jsonl"
            selected = root / "selected.json"
            candidate = root / "candidates.pt"
            output = root / "selection_manifest.json"
            rows = [
                {
                    "start_state_id": "source-0",
                    "command_mode": "pure_x",
                    "command_linear_active": True,
                    "command_yaw_active": False,
                    "command_vx_mean": 0.3,
                    "command_vy_mean": 0.0,
                    "command_yaw_mean": 0.0,
                    "command_linear_speed_mean": 0.3,
                    "minimum_active_axis_response_ratio": 0.8,
                    "minimum_active_axis_transport_ratio": 0.7,
                    "linear_tracking_error_mean": 0.05,
                    "linear_tracking_error_steady": 0.04,
                    "command_direction_correct_fraction": 0.95,
                    "command_direction_violation_rate": 0.05,
                    "survival_fraction": 1.0,
                    "tilt_mean": 0.05,
                    "roll_pitch_rate_rms": 0.2,
                    "action_delta_norm_mean": 0.2,
                    "action_saturation_fraction": 0.0,
                    "trace_selection_eligible": True,
                },
                {
                    "start_state_id": "source-1",
                    "command_mode": "stand",
                    "command_linear_active": False,
                    "command_yaw_active": False,
                    "linear_tracking_error_mean": 0.005,
                    "yaw_tracking_error_mean": 0.005,
                    "survival_fraction": 1.0,
                    "tilt_mean": 0.02,
                    "roll_pitch_rate_rms": 0.05,
                    "action_delta_norm_mean": 0.05,
                    "action_saturation_fraction": 0.0,
                    "trace_selection_eligible": True,
                },
            ]
            summaries.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            candidate.write_bytes(b"synthetic-candidate-dataset")
            selected.write_text(
                json.dumps({"selected_indices": [0]}),
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    "trace-rule-score",
                    "--summaries",
                    str(summaries),
                    "--scores-output",
                    str(scores),
                    "--manifest-output",
                    str(score_manifest),
                    "--scored-summaries-output",
                    str(scored_summaries),
                ],
            ):
                score_main()
            with patch.object(
                sys,
                "argv",
                [
                    "trace-export-selection",
                    "--summaries",
                    str(summaries),
                    "--scores",
                    str(scores),
                    "--score-manifest",
                    str(score_manifest),
                    "--candidate-dataset",
                    str(candidate),
                    "--selected-indices-json",
                    str(selected),
                    "--output",
                    str(output),
                ],
            ):
                export_main()
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "portable_trace_selection_v1")
            self.assertEqual(manifest["score_source"]["kind"], "rule")
            self.assertEqual(manifest["selected_indices"], [0])
            self.assertEqual(
                manifest["selected_identities"][0]["start_state_id"],
                "source-0",
            )


if __name__ == "__main__":
    unittest.main()
