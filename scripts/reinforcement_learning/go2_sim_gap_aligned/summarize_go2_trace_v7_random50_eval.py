#!/usr/bin/env python3
"""Summarize paired fixed-horizon V7 evaluations with bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "mean_return", "fixed_horizon_return", "terminated_count", "termination_rate",
    "mean_episode_length", "survival_at_50", "survival_at_100", "survival_at_200",
    "survival_at_500", "survival_at_1000", "survive_to_1000_ratio",
    "restricted_mean_time_to_first_failure", "mean_resets_per_env",
    "terminations_per_1000_env_steps", "reward_per_alive_step", "error_vel_xy",
    "error_vel_yaw", "post_switch_error_vel_xy_10", "post_switch_error_vel_yaw_10",
    "roll_abs_mean", "pitch_abs_mean", "tilt_mean", "action_saturation_ratio",
    "action_delta_abs_mean", "epistemic_uncertainty",
)


def bootstrap_ci(values: list[float], seed: int = 20260719) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(20_000, len(data)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.eval_root.glob("gap_only/*/*/*/seed_*.json")):
        _, side, condition, variant, _ = path.relative_to(args.eval_root).parts
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"side": side, "condition": condition, "variant": variant,
               "seed": int(data["seed"]), "path": str(path.resolve())}
        row.update({metric: float(data[metric]) for metric in METRICS})
        row["command_sequence"] = data["command_sequence"]
        rows.append(row)
    if not rows:
        raise SystemExit("No completed evaluations")

    commands = {}
    for row in rows:
        key = row["seed"]
        if key in commands and commands[key] != row["command_sequence"]:
            raise ValueError(f"Command mismatch for seed {key}")
        commands[key] = row["command_sequence"]

    groups = defaultdict(list)
    for row in rows:
        groups[(row["side"], row["condition"], row["variant"])].append(row)
    summary = []
    for (side, condition, variant), group in sorted(groups.items()):
        out = {"side": side, "condition": condition, "variant": variant,
               "num_seeds": len(group)}
        for metric in METRICS:
            values = np.asarray([item[metric] for item in group], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std())
        baseline = {item["seed"]: item for item in groups.get((side, condition, "baseline"), [])}
        paired = [item for item in group if item["seed"] in baseline]
        if variant != "baseline" and paired:
            for metric in METRICS:
                deltas = [item[metric] - baseline[item["seed"]][metric] for item in paired]
                lo, hi = bootstrap_ci(deltas)
                out[f"{metric}_paired_delta"] = float(np.mean(deltas))
                out[f"{metric}_paired_delta_ci95_low"] = lo
                out[f"{metric}_paired_delta_ci95_high"] = hi
        summary.append(out)

    def write_csv(path: Path, data: list[dict]) -> None:
        fields = sorted({key for row in data for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)

    args.eval_root.mkdir(parents=True, exist_ok=True)
    raw = [{k: v for k, v in row.items() if k != "command_sequence"} for row in rows]
    write_csv(args.eval_root / "per_seed_metrics.csv", raw)
    write_csv(args.eval_root / "paired_summary.csv", summary)
    (args.eval_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    km_rows = []
    for row in rows:
        source = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        for time, survival in zip(
            source["kaplan_meier"]["time"], source["kaplan_meier"]["survival"]
        ):
            km_rows.append({
                "side": row["side"], "condition": row["condition"],
                "variant": row["variant"], "seed": row["seed"],
                "time": time, "survival": survival,
            })
    write_csv(args.eval_root / "kaplan_meier.csv", km_rows)
    lines = ["# Corrected V7 gap-only random50 evaluation", "",
             "256 envs, 1000 steps, command switch every 50 steps, seeds 400/401/402.", "",
             "| Side | Gap | Method | N | Return | Term rate | Episode length | Survive | XY | Yaw | XY@10 | Yaw@10 | Uncertainty |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['side']} | {row['condition']} | {row['variant']} | {row['num_seeds']} | "
            f"{row['mean_return_mean']:.3f} | {row['termination_rate_mean']:.4f} | "
            f"{row['mean_episode_length_mean']:.1f} | {row['survive_to_1000_ratio_mean']:.4f} | "
            f"{row['error_vel_xy_mean']:.4f} | {row['error_vel_yaw_mean']:.4f} | "
            f"{row['post_switch_error_vel_xy_10_mean']:.4f} | "
            f"{row['post_switch_error_vel_yaw_10_mean']:.4f} | "
            f"{row['epistemic_uncertainty_mean']:.4f} |"
        )
    (args.eval_root / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
