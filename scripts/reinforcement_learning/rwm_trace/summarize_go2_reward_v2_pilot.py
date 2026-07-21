#!/usr/bin/env python3
"""Compare paired fresh-buffer V1 and hierarchical-V2 RWM-only pilots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _mean(rows: list[dict[str, Any]], field: str) -> float:
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
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.run_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    results = {
        arm: _metrics(root / arm / "behavior" / "summary.json")
        for arm in ("reward_v1", "reward_v2")
    }
    report = {
        "schema": "go2_trace_v10_reward_v2_pilot_comparison_v1",
        "run_root": str(root),
        "config": str(config_path),
        "config_schema": config["schema"],
        "results": results,
        "v2_minus_v1": {
            key: float(results["reward_v2"][key] - results["reward_v1"][key])
            for key in results["reward_v2"]
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
