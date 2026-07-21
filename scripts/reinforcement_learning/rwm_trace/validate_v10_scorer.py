"""Fail-closed V10 scorer quality and seven-mode anti-collapse validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.scorer import load_scorer_checkpoint, score_summaries
from scripts.reinforcement_learning.rwm_trace.v10_protocol import COMMAND_MODES


MOTION_MODES = COMMAND_MODES[1:]


def _template(mode: str, *, moving: bool) -> dict:
    linear = "x" in mode or mode in {"pure_x", "pure_y", "xy", "y_yaw", "xy_yaw"}
    yaw = "yaw" in mode
    row = {
        "command_mode": mode,
        **{f"command_mode_{name}": float(name == mode) for name in COMMAND_MODES},
        "command_linear_active": linear,
        "command_yaw_active": yaw,
        "survival_fraction": 0.92 if moving else 1.0,
        "terminal_flag": False,
        "tilt_mean": 0.04 if moving else 0.015,
        "tilt_max": 0.08 if moving else 0.03,
        "action_saturation_rate": 0.01 if moving else 0.0,
        "command_direction_correct_fraction": 0.90 if moving else 0.50,
        "command_direction_violation_rate": 0.10 if moving else 0.50,
        "linear_velocity_realization_ratio_mean": 0.45 if moving and linear else (0.01 if linear else float("nan")),
        "linear_velocity_realization_ratio_steady": 0.45 if moving and linear else (0.01 if linear else float("nan")),
        "yaw_velocity_realization_ratio_mean": 0.45 if moving and yaw else (0.01 if yaw else float("nan")),
        "yaw_velocity_realization_ratio_steady": 0.45 if moving and yaw else (0.01 if yaw else float("nan")),
        "command_displacement_realization_ratio": 0.45 if moving and linear else (0.01 if linear else float("nan")),
        "yaw_displacement_realization_ratio": 0.45 if moving and yaw else (0.01 if yaw else float("nan")),
        "linear_tracking_error_mean": 0.10 if moving and linear else (0.35 if linear else 0.0),
        "linear_tracking_error_steady": 0.10 if moving and linear else (0.35 if linear else 0.0),
        "yaw_tracking_error_mean": 0.08 if moving and yaw else (0.30 if yaw else 0.0),
        "yaw_tracking_error_steady": 0.08 if moving and yaw else (0.30 if yaw else 0.0),
        "command_projected_velocity_window": 0.15 if moving and linear else 0.0,
        "yaw_velocity_window": 0.15 if moving and yaw else 0.0,
        "base_speed_mean": 0.18 if moving else 0.01,
        "contact_switch_rate": 0.12 if moving else 0.0,
    }
    for foot in ("fr", "fl", "rr", "rl"):
        row[f"foot_swing_rate_{foot}"] = 1.5 if moving else 0.0
    row["longest_stance_fraction_rr"] = 0.55 if moving else 1.0
    row["longest_stance_fraction_rl"] = 0.55 if moving else 1.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-global-balanced-accuracy", type=float, default=0.60)
    parser.add_argument("--minimum-mode-balanced-accuracy", type=float, default=0.55)
    parser.add_argument("--minimum-mode-validation-count", type=int, default=12)
    args = parser.parse_args()
    model, stats, metadata = load_scorer_checkpoint(args.checkpoint)
    failures: list[str] = []
    global_accuracy = float(metadata.get("val_balanced_accuracy", float("nan")))
    if not math.isfinite(global_accuracy) or global_accuracy < args.minimum_global_balanced_accuracy:
        failures.append("global_balanced_accuracy")
    mode_metrics = dict(metadata.get("val_per_command_mode") or {})
    for mode in COMMAND_MODES:
        metric = dict(mode_metrics.get(mode) or {})
        accuracy = float(metric.get("balanced_accuracy", float("nan")))
        if (
            int(metric.get("count", 0)) < args.minimum_mode_validation_count
            or int(metric.get("count_i", 0)) < 1
            or int(metric.get("count_j", 0)) < 1
            or not math.isfinite(accuracy)
            or accuracy < args.minimum_mode_balanced_accuracy
        ):
            failures.append(f"mode:{mode}")
    anti_collapse = []
    for mode in MOTION_MODES:
        moving, collapsed = _template(mode, moving=True), _template(mode, moving=False)
        scores = score_summaries(model, [moving, collapsed], stats)
        passed = bool(float(scores[0]) > float(scores[1]))
        anti_collapse.append({
            "mode": mode,
            "moving_score": float(scores[0]),
            "collapsed_score": float(scores[1]),
            "passed": passed,
        })
        if not passed:
            failures.append(f"anti_collapse:{mode}")
    report = {
        "schema": "go2_trace_v10_scorer_validation_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "global_balanced_accuracy": global_accuracy,
        "anti_collapse_pairs": anti_collapse,
        "failures": failures,
        "passed": not failures,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"V10 scorer validation failed: {failures}")


if __name__ == "__main__":
    main()
