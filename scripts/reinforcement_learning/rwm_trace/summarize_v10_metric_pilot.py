#!/usr/bin/env python3
"""Compare paired no-TRACE, random, and metric V10 pilot evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else float("nan")


def _metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clean = payload["clean"]
    nonstand = clean["per_command"][1:]
    return {
        "all_nonzero_commands_pass_rate": float(clean["all_nonzero_commands_pass_rate_mean"]),
        "mean_no_response_rate": float(clean["mean_no_response_rate"]),
        "mean_nonstand_termination_rate": _mean(nonstand, "termination_rate_mean"),
        "mean_nonstand_xy_error": _mean(nonstand, "error_vel_xy_mean"),
        "mean_nonstand_yaw_error": _mean(nonstand, "error_vel_yaw_mean"),
        "mean_nonstand_direction_correct_fraction": _mean(
            nonstand, "direction_correct_fraction_mean"
        ),
        "mean_nonstand_effective_swing_fraction": _mean(
            nonstand, "effective_swing_foot_fraction_mean"
        ),
        "mean_nonstand_action_saturation_rate": _mean(
            nonstand, "action_saturation_rate_mean"
        ),
    }


def main() -> None:
    args = _parse_args()
    root = Path(args.run_root).expanduser().resolve()
    results = {
        branch: _metrics(root / branch / "behavior" / "summary.json")
        for branch in ("control", "random", "metric")
    }
    report = {
        "schema": "go2_trace_v10_metric_pilot_comparison_v1",
        "run_root": str(root),
        "results": results,
        "metric_minus_control": {
            key: float(results["metric"][key] - results["control"][key])
            for key in results["metric"]
        },
        "metric_minus_random": {
            key: float(results["metric"][key] - results["random"][key])
            for key in results["metric"]
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
