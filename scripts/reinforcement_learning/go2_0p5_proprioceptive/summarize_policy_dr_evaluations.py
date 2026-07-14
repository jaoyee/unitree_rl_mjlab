"""Aggregate per-seed real-MJLab policy evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRICS = (
    "mean_return",
    "mean_episode_length",
    "terminated_count",
    "error_vel_xy",
    "error_vel_yaw",
    "action_abs_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--eval_root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.eval_root.glob("*__seed*.json")):
        condition = path.name.split("__seed", maxsplit=1)[0]
        grouped[condition].append(json.loads(path.read_text(encoding="utf-8")))
    if not grouped:
        raise FileNotFoundError(f"No evaluation JSON files found in {args.eval_root}")

    summaries: list[dict[str, Any]] = []
    for condition, records in grouped.items():
        row: dict[str, Any] = {
            "condition": condition,
            "seeds": [int(record["seed"]) for record in records],
            "num_runs": len(records),
        }
        for metric in METRICS:
            values = [float(record[metric]) for record in records]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        denom = sum(int(record["num_envs"]) * int(record["steps"]) for record in records)
        row["termination_per_1000_env_steps"] = (
            1000.0 * sum(int(record["terminated_count"]) for record in records) / max(denom, 1)
        )
        summaries.append(row)

    summaries.sort(key=lambda row: row["condition"])
    output = {
        "eval_root": str(args.eval_root.resolve()),
        "conditions": summaries,
    }
    (args.eval_root / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    fieldnames = list(summaries[0].keys())
    with (args.eval_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({**row, "seeds": ",".join(map(str, row["seeds"]))})

    lines = [
        "# Policy DR evaluation summary",
        "",
        "| Condition | Return | Episode length | Terminations / 1000 env-steps | Velocity error XY | Yaw error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['mean_return_mean']:.3f} | "
            f"{row['mean_episode_length_mean']:.2f} | "
            f"{row['termination_per_1000_env_steps']:.4f} | "
            f"{row['error_vel_xy_mean']:.4f} | {row['error_vel_yaw_mean']:.4f} |"
        )
    (args.eval_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
