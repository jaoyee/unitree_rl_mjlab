#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = [
    "mean_return",
    "termination_rate",
    "mean_episode_length_with_right_censoring",
    "error_vel_xy",
    "error_vel_yaw",
    "epistemic_uncertainty",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    root = args.eval_root / "refresh_curves" / "sim"
    for path in sorted(root.glob("*/*/refresh_*/seed_*.json")):
        rel = path.relative_to(root).parts
        condition, variant, refresh_name, _ = rel
        payload = json.loads(path.read_text())
        row = {
            "condition": condition,
            "variant": variant,
            "refresh": int(refresh_name.split("_")[-1]),
            "seed": int(path.stem.split("_")[-1]),
        }
        row["cumulative_policy_steps"] = 10_000_000 + row["refresh"] * 5_000_000
        for metric in METRICS:
            row[metric] = payload.get(metric)
        rows.append(row)

    args.eval_root.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (args.eval_root / "refresh_per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    keys = sorted({(row["condition"], row["variant"], row["refresh"]) for row in rows})
    for condition, variant, refresh in keys:
        group = [r for r in rows if (r["condition"], r["variant"], r["refresh"]) == (condition, variant, refresh)]
        record = {
            "condition": condition,
            "variant": variant,
            "refresh": refresh,
            "cumulative_policy_steps": group[0]["cumulative_policy_steps"],
            "n_seeds": len(group),
        }
        for metric in METRICS:
            values = np.asarray([r[metric] for r in group if r[metric] is not None], dtype=float)
            record[f"{metric}_mean"] = float(values.mean()) if values.size else None
            record[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else None
        aggregate.append(record)
    fields = sorted({key for row in aggregate for key in row})
    with (args.eval_root / "refresh_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)

    lines = [
        "# Corrected V7 refresh curves",
        "",
        "Protocol: same-gap, 256 environments, 1000 steps, commands switch every 50 steps, seeds 400/401/402.",
        "",
        "|condition|variant|refresh|steps|return|termination rate|episode length|XY error|yaw error|uncertainty|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        def fmt(key):
            value = row.get(key)
            return "NA" if value is None else f"{value:.4f}"
        lines.append(
            f"|{row['condition']}|{row['variant']}|{row['refresh']}|{row['cumulative_policy_steps']}|"
            f"{fmt('mean_return_mean')}|{fmt('termination_rate_mean')}|"
            f"{fmt('mean_episode_length_with_right_censoring_mean')}|{fmt('error_vel_xy_mean')}|"
            f"{fmt('error_vel_yaw_mean')}|{fmt('epistemic_uncertainty_mean')}|"
        )
    (args.eval_root / "refresh_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
