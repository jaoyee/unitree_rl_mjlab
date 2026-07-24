#!/usr/bin/env python3
"""Attach the complete validated TRACE eligibility gate to trajectory summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _finite(row: dict, name: str, default: float) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _active_flags(row: dict) -> tuple[bool, bool]:
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


def _motion_rejection(row: dict, validity: dict) -> str | None:
    linear, yaw = _active_flags(row)
    if not (linear or yaw):
        return None
    minimum_component = float(
        validity.get(
            "minimum_component_realization_ratio",
            min(
                validity.get("minimum_linear_realization_ratio", 0.1),
                validity.get("minimum_yaw_realization_ratio", 0.1),
            ),
        )
    )
    explicit_response = row.get("minimum_active_axis_response_ratio")
    if explicit_response is not None:
        if _finite(row, "minimum_active_axis_response_ratio", -math.inf) < minimum_component:
            return "insufficient_component_response"
    elif not bool(row.get("command_motion_gate_passed", False)):
        return str(
            row.get("command_motion_gate_rejection_reason")
            or "command_motion_gate"
        )
    explicit_transport = row.get("minimum_active_axis_transport_ratio")
    if (
        explicit_transport is not None
        and _finite(row, "minimum_active_axis_transport_ratio", -math.inf) <= 0.0
    ):
        return "nonpositive_component_transport"
    if (
        _finite(row, "command_direction_violation_rate", 1.0)
        > float(validity["maximum_direction_violation_rate"])
    ):
        return "direction_violation"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    validity = protocol["validity"]
    max_reset_error = float(validity["maximum_reset_reconstruction_error"])
    max_saturation = float(validity["maximum_saturation_fraction"])
    rows = [
        json.loads(line)
        for line in Path(args.summaries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: dict[str, int] = {}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            reason = None
            reset_error = float(row.get("reset_reconstruction_error", float("nan")))
            saturation = float(row.get("action_saturation_fraction", float("inf")))
            if bool(row.get("nonfinite_flag", True)):
                reason = "nonfinite"
            elif not math.isfinite(reset_error):
                reason = "reset_reconstruction_error_nonfinite"
            elif reset_error > max_reset_error:
                reason = "reset_reconstruction_error"
            elif saturation > max_saturation:
                reason = "action_saturation"
            else:
                reason = _motion_rejection(row, validity)
            row["trace_selection_eligible"] = reason is None
            row["trace_selection_rejection_reason"] = reason
            if reason is not None:
                counts[reason] = counts.get(reason, 0) + 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "schema": "portable_trace_eligibility_v1",
                "row_count": len(rows),
                "eligible_count": len(rows) - sum(counts.values()),
                "rejection_counts": counts,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
