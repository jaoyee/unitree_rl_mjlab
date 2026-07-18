"""Choose one global real contact threshold against fixed sim contact references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SIM_ALL_FOUR_REFERENCE = {"g0": 0.34, "rr05": 0.38, "rr03": 0.41, "p5": 0.52, "p75": 0.60}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summaries = {}
    for raw in args.report:
        payload = json.loads(Path(raw).read_text())
        for condition, by_threshold in payload["conditions"].items():
            for threshold, values in by_threshold.items():
                row = summaries.setdefault(threshold, {"conditions": {}})
                row["conditions"][condition] = {
                    "real_all_four": values["all_four_contact_fraction"],
                    "sim_all_four_reference": SIM_ALL_FOUR_REFERENCE[condition],
                    "absolute_contact_rate_error": abs(
                        values["all_four_contact_fraction"] - SIM_ALL_FOUR_REFERENCE[condition]
                    ),
                    "zero_contact_fraction": values["zero_contact_fraction"],
                    "base_velocity_xy_command_correlation": values["base_velocity_command_correlation"][:2],
                    "base_velocity_confidence_mean": values["base_velocity_confidence_mean"],
                    "stand_base_velocity_xy_abs_mean": values["stand_base_velocity_abs_mean"][:2],
                }
    for row in summaries.values():
        conditions = list(row["conditions"].values())
        row["contact_rate_mae"] = sum(item["absolute_contact_rate_error"] for item in conditions) / len(conditions)
        correlations = [value for item in conditions for value in item["base_velocity_xy_command_correlation"] if value is not None]
        row["mean_xy_command_correlation"] = sum(correlations) / len(correlations)
        row["mean_velocity_confidence"] = sum(item["base_velocity_confidence_mean"] for item in conditions) / len(conditions)
        row["maximum_zero_contact_fraction"] = max(item["zero_contact_fraction"] for item in conditions)
    eligible = {
        threshold: row for threshold, row in summaries.items()
        if row["maximum_zero_contact_fraction"] < 0.01
    }
    selected = min(eligible, key=lambda threshold: (
        eligible[threshold]["contact_rate_mae"],
        -eligible[threshold]["mean_xy_command_correlation"],
        -eligible[threshold]["mean_velocity_confidence"],
    ))
    output = {
        "schema": "go2_global_contact_threshold_decision_v2",
        "selected_threshold_newton": float(selected),
        "selection_rule": "minimum five-condition all-four-contact MAE to fixed sim references; require zero-contact <1%",
        "sim_all_four_contact_reference": SIM_ALL_FOUR_REFERENCE,
        "thresholds": summaries,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_threshold_newton": float(selected),
        "scores": {key: {
            "contact_rate_mae": value["contact_rate_mae"],
            "mean_xy_command_correlation": value["mean_xy_command_correlation"],
            "mean_velocity_confidence": value["mean_velocity_confidence"],
        } for key, value in summaries.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
