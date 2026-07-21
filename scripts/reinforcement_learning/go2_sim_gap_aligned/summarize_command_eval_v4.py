#!/usr/bin/env python3
"""Aggregate multi-seed clean/calibrated command-response evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    report: dict[str, object] = {"schema": "go2_command_eval_v4_summary", "input_dir": str(input_dir)}
    for mode in ("clean", "calibrated"):
        payloads = [json.loads(path.read_text()) for path in sorted(input_dir.glob(f"{mode}_seed*.json"))]
        if not payloads:
            continue
        command_count = len(payloads[0]["command_metrics_v4"]["per_command"])
        per_command = []
        for index in range(command_count):
            entries = [payload["command_metrics_v4"]["per_command"][index] for payload in payloads]
            per_command.append(
                {
                    "index": index,
                    "command": entries[0]["command"],
                    "pass_rate_mean": float(np.mean([entry["pass_rate"] for entry in entries])),
                    "pass_rate_min_seed": float(np.min([entry["pass_rate"] for entry in entries])),
                    "no_response_rate_mean": float(
                        np.mean([entry["no_response_rate"] for entry in entries])
                    ),
                    "termination_rate_mean": float(
                        np.mean([entry["termination_rate"] for entry in entries])
                    ),
                    "error_vel_xy_mean": float(
                        np.mean([entry["mean_error_vel_xy"] for entry in entries])
                    ),
                    "error_vel_yaw_mean": float(
                        np.mean([entry["mean_error_vel_yaw"] for entry in entries])
                    ),
                    "action_saturation_rate_mean": float(
                        np.mean([entry["action_saturation_transition_rate"] for entry in entries])
                    ),
                    "linear_response_gain_mean": float(
                        np.mean([entry["linear_response_gain_mean"] for entry in entries])
                    ),
                    "linear_response_gain_median_mean": float(
                        np.mean([entry["linear_response_gain_median"] for entry in entries])
                    ),
                    "yaw_response_gain_mean": float(
                        np.mean([entry["yaw_response_gain_mean"] for entry in entries])
                    ),
                    "yaw_response_gain_median_mean": float(
                        np.mean([entry["yaw_response_gain_median"] for entry in entries])
                    ),
                    "direction_correct_fraction_mean": float(
                        np.mean([entry["direction_correct_fraction"] for entry in entries])
                    ),
                    "effective_swing_foot_fraction_mean": (
                        float(np.mean([entry["effective_swing_foot_fraction_mean"] for entry in entries]))
                        if entries[0].get("effective_swing_foot_fraction_mean") is not None
                        else None
                    ),
                    "no_effective_swing_rate_mean": (
                        float(np.mean([entry["no_effective_swing_rate"] for entry in entries]))
                        if entries[0].get("no_effective_swing_rate") is not None
                        else None
                    ),
                }
            )
        aggregate = [payload["command_metrics_v4"] for payload in payloads]
        report[mode] = {
            "seeds": [int(payload["seed"]) for payload in payloads],
            "mean_segment_pass_rate": float(
                np.mean([entry["mean_segment_pass_rate"] for entry in aggregate])
            ),
            "minimum_segment_pass_rate": float(
                np.min([entry["minimum_segment_pass_rate"] for entry in aggregate])
            ),
            "all_commands_pass_rate_mean": float(
                np.mean([entry["all_commands_pass_rate"] for entry in aggregate])
            ),
            "all_nonzero_commands_pass_rate_mean": float(
                np.mean([entry["all_nonzero_commands_pass_rate"] for entry in aggregate])
            ),
            "mean_no_response_rate": float(
                np.mean([entry["mean_no_response_rate"] for entry in aggregate])
            ),
            "mean_effective_swing_foot_fraction_nonstand": (
                float(np.mean([entry["mean_effective_swing_foot_fraction_nonstand"] for entry in aggregate]))
                if aggregate[0].get("mean_effective_swing_foot_fraction_nonstand") is not None
                else None
            ),
            "mean_no_effective_swing_rate_nonstand": (
                float(np.mean([entry["mean_no_effective_swing_rate_nonstand"] for entry in aggregate]))
                if aggregate[0].get("mean_no_effective_swing_rate_nonstand") is not None
                else None
            ),
            "per_command": per_command,
        }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
