#!/usr/bin/env python3
"""Behavior-first rule selector used to validate TRACE before a learned scorer.

Eligibility is the hard first tier.  For an active command, eligibility means
that the branch actually responds to every commanded component.  Only within
that tier do tracking/response (70%) and motion stability (30%) trade off.
Thus stability means stable motion, not standing still under a motion command.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    identity,
    read_jsonl,
    sha256_path,
)


def _finite(row: dict[str, Any], name: str, default: float) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _quality_from_error(error: float, scale: float) -> float:
    return 1.0 / (1.0 + max(0.0, error) / max(scale, 1.0e-6))


def _active_flags(row: dict[str, Any]) -> tuple[bool, bool]:
    linear = bool(
        row.get(
            "command_linear_active",
            math.hypot(
                _finite(row, "command_vx_mean", 0.0),
                _finite(row, "command_vy_mean", 0.0),
            )
            > 0.02,
        )
    )
    yaw = bool(
        row.get(
            "command_yaw_active",
            abs(_finite(row, "command_yaw_mean", 0.0)) > 0.02,
        )
    )
    return linear, yaw


def _component_response(row: dict[str, Any], linear: bool, yaw: bool) -> float:
    explicit = row.get("minimum_active_axis_response_ratio")
    if explicit is not None:
        return _clip01(_finite(row, "minimum_active_axis_response_ratio", 0.0))
    values: list[float] = []
    if linear:
        values.append(
            min(
                _finite(row, "linear_velocity_realization_ratio_mean", 0.0),
                _finite(row, "linear_velocity_realization_ratio_steady", 0.0),
            )
        )
    if yaw:
        values.append(
            min(
                _finite(row, "yaw_velocity_realization_ratio_mean", 0.0),
                _finite(row, "yaw_velocity_realization_ratio_steady", 0.0),
            )
        )
    return _clip01(min(values) if values else 0.0)


def _component_transport(row: dict[str, Any], linear: bool, yaw: bool) -> float:
    explicit = row.get("minimum_active_axis_transport_ratio")
    if explicit is not None:
        return _clip01(_finite(row, "minimum_active_axis_transport_ratio", 0.0))
    values: list[float] = []
    if linear:
        values.append(_finite(row, "command_displacement_realization_ratio", 0.0))
    if yaw:
        values.append(_finite(row, "yaw_displacement_realization_ratio", 0.0))
    return _clip01(min(values) if values else 0.0)


def _tracking_quality(row: dict[str, Any], linear: bool, yaw: bool) -> float:
    values: list[float] = []
    if linear:
        command_scale = max(
            _finite(row, "command_linear_speed_mean", 0.0),
            0.05,
        )
        error = max(
            _finite(row, "linear_tracking_error_mean", float("inf")),
            _finite(row, "linear_tracking_error_steady", float("inf")),
        )
        values.append(_quality_from_error(error, command_scale))
    if yaw:
        command_scale = max(abs(_finite(row, "command_yaw_mean", 0.0)), 0.05)
        error = max(
            _finite(row, "yaw_tracking_error_mean", float("inf")),
            _finite(row, "yaw_tracking_error_steady", float("inf")),
        )
        values.append(_quality_from_error(error, command_scale))
    return min(values) if values else 0.0


def _stability_quality(row: dict[str, Any]) -> float:
    survival = _clip01(_finite(row, "survival_fraction", 0.0))
    tilt = math.exp(-max(0.0, _finite(row, "tilt_mean", float("inf"))) / 0.20)
    rate = math.exp(
        -max(0.0, _finite(row, "roll_pitch_rate_rms", float("inf"))) / 1.0
    )
    action_delta = math.exp(
        -max(0.0, _finite(row, "action_delta_norm_mean", float("inf"))) / 1.0
    )
    saturation = 1.0 - _clip01(
        _finite(row, "action_saturation_fraction", 1.0)
    )
    return float(np.mean([survival, tilt, rate, action_delta, saturation]))


def rule_components(row: dict[str, Any]) -> dict[str, float | bool | str]:
    linear, yaw = _active_flags(row)
    active = linear or yaw
    eligible = bool(row.get("trace_selection_eligible", False))
    stability = _stability_quality(row)
    if not active:
        linear_error = _finite(row, "linear_tracking_error_mean", float("inf"))
        yaw_error = _finite(row, "yaw_tracking_error_mean", float("inf"))
        stationary = min(
            _quality_from_error(linear_error, 0.05),
            _quality_from_error(yaw_error, 0.05),
        )
        primary = stationary
        profile = "stand"
    else:
        response = _component_response(row, linear, yaw)
        transport = _component_transport(row, linear, yaw)
        tracking = _tracking_quality(row, linear, yaw)
        direction = min(
            _clip01(_finite(row, "command_direction_correct_fraction", 0.0)),
            1.0
            - _clip01(_finite(row, "command_direction_violation_rate", 1.0)),
        )
        # The weakest commanded component controls the active-command score.
        primary = min(response, transport, tracking, direction)
        profile = "active"
    # The two-point eligibility tier is larger than the full [0, 1] weighted
    # objective. An ineligible, perfectly stable branch therefore cannot
    # outrank any eligible branch.
    score = float(
        (2.0 if eligible else 0.0)
        + 0.70 * primary
        + 0.30 * stability
    )
    return {
        "profile": profile,
        "eligible": eligible,
        "primary_quality": float(primary),
        "stability_quality": float(stability),
        "rule_score": score,
        "linear_active": linear,
        "yaw_active": yaw,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--scored-summaries-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries_path = Path(args.summaries).expanduser().resolve()
    rows = read_jsonl(summaries_path)
    score_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        components = rule_components(row)
        score = float(components["rule_score"])
        scores.append(score)
        score_rows.append(
            {
                **identity(row, index),
                "score": score,
                "rule_components": components,
            }
        )
        scored_rows.append(
            {
                **row,
                "trace_score": score,
                "trace_rule_components": components,
            }
        )
    atomic_write_jsonl(args.scores_output, score_rows)
    atomic_write_jsonl(args.scored_summaries_output, scored_rows)
    score_array = np.asarray(scores, dtype=np.float32)
    manifest = {
        "schema": "portable_trace_rule_scores_v1",
        "score_source": {
            "kind": "rule",
            "profile": "eligible_motion_then_tracking70_stability30_v1",
            "tracking_response_weight": 0.70,
            "motion_stability_weight": 0.30,
            "eligibility_tier_offset": 2.0,
        },
        "summaries_path": str(summaries_path),
        "summaries_sha256": sha256_path(summaries_path),
        "scores_path": str(Path(args.scores_output).expanduser().resolve()),
        "scores_sha256": sha256_path(args.scores_output),
        "row_count": len(rows),
        "score_statistics": {
            "min": float(score_array.min()) if len(score_array) else None,
            "max": float(score_array.max()) if len(score_array) else None,
            "mean": float(score_array.mean()) if len(score_array) else None,
            "std": float(score_array.std()) if len(score_array) else None,
        },
    }
    atomic_write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
