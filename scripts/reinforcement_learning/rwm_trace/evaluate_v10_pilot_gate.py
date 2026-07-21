"""Compare one-refresh TRACE against its paired no-TRACE control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.v10_protocol import load_protocol


def _active(command: list[float]) -> tuple[bool, bool]:
    return bool(np.linalg.norm(command[:2]) > 1.0e-6), bool(abs(command[2]) > 1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--trace-summary", required=True)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol, _, _ = load_protocol(args.protocol)
    limits = protocol["pilot"]
    trace = json.loads(Path(args.trace_summary).read_text(encoding="utf-8"))["clean"]
    control = json.loads(Path(args.control_summary).read_text(encoding="utf-8"))["clean"]
    trace_rows = trace["per_command"]
    control_rows = control["per_command"]
    if len(trace_rows) != 8 or len(control_rows) != 8:
        raise ValueError("V10 pilot evaluation must contain all eight independent command modes.")
    failures: list[str] = []
    mode_checks = []
    trace_errors: list[float] = []
    control_errors: list[float] = []
    for index in range(1, 8):
        current, baseline = trace_rows[index], control_rows[index]
        command = current["command"]
        linear, yaw = _active(command)
        checks = {
            "direction": float(current["direction_correct_fraction_mean"])
            >= float(limits["minimum_direction_correct_fraction"]),
        }
        if linear:
            checks["linear_realization"] = float(current["linear_response_gain_median_mean"]) >= float(
                limits["minimum_component_realization"]
            )
            trace_errors.append(float(current["error_vel_xy_mean"]))
            control_errors.append(float(baseline["error_vel_xy_mean"]))
        if yaw:
            checks["yaw_realization"] = float(current["yaw_response_gain_median_mean"]) >= float(
                limits["minimum_component_realization"]
            )
            trace_errors.append(float(current["error_vel_yaw_mean"]))
            control_errors.append(float(baseline["error_vel_yaw_mean"]))
        if not all(checks.values()):
            failures.append(f"command_{index}")
        mode_checks.append({"index": index, "command": command, "checks": checks})
    trace_error = float(np.mean(trace_errors))
    control_error = float(np.mean(control_errors))
    tracking_degradation = (trace_error - control_error) / max(control_error, 1.0e-8)
    trace_survival = 1.0 - float(np.mean([row["termination_rate_mean"] for row in trace_rows[1:]]))
    control_survival = 1.0 - float(np.mean([row["termination_rate_mean"] for row in control_rows[1:]]))
    survival_drop = control_survival - trace_survival
    if tracking_degradation > float(limits["maximum_tracking_error_degradation_fraction"]):
        failures.append("tracking_degradation")
    if survival_drop > float(limits["maximum_survival_drop"]):
        failures.append("survival_drop")
    report = {
        "schema": "go2_trace_v10_pilot_gate_v1",
        "mode_checks": mode_checks,
        "trace_tracking_error": trace_error,
        "control_tracking_error": control_error,
        "tracking_error_degradation_fraction": tracking_degradation,
        "trace_survival": trace_survival,
        "control_survival": control_survival,
        "survival_drop": survival_drop,
        "failures": failures,
        "passed": not failures,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"V10 pilot gate failed: {failures}")


if __name__ == "__main__":
    main()
