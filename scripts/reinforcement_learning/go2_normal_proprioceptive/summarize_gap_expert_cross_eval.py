#!/usr/bin/env python3
"""Summarize payload and RR-calf expert cross-evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


FAMILIES = {
    "payload": {
        "expected": 24,
        "nominal": "M0",
        "hardest": "M10",
        "order": {"M0": 0, "M5": 1, "M7p5": 2, "M10": 3},
    },
    "rr_calf": {
        "expected": 18,
        "nominal": "S1p0",
        "hardest": "S0p5",
        "order": {"S1p0": 0, "S0p75": 1, "S0p5": 2},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--rr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_family(family: str, root: Path, steps: int) -> list[dict]:
    rows = []
    for path in sorted((root / "cross_eval").glob("G*_D*/*/metrics.json")):
        train_id = path.parent.parent.name
        eval_id = path.parent.name
        metrics = json.loads(path.read_text(encoding="utf-8"))
        episode_length = float(metrics["mean_episode_length"])
        rows.append({
            "family": family,
            "train_id": train_id,
            "gap_train": train_id.split("_", 1)[0],
            "dr": train_id.split("_", 1)[1],
            "eval_id": eval_id,
            "eval_order": FAMILIES[family]["order"][eval_id],
            "mean_step_reward": float(metrics["mean_step_reward"]),
            "mean_episode_length": episode_length,
            "horizon_survival": min(episode_length / steps, 1.0),
            "non_timeout_terminations": int(metrics["non_timeout_termination_count"]),
            "error_vel_xy": float(metrics["error_vel_xy"]),
            "error_vel_yaw": float(metrics["error_vel_yaw"]),
            "action_abs_mean": float(metrics["action_abs_mean"]),
            "metrics_path": str(path.resolve()),
        })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    summaries = []
    keys = sorted({(row["family"], row["train_id"]) for row in rows})
    for family, train_id in keys:
        group = sorted(
            (row for row in rows if row["family"] == family and row["train_id"] == train_id),
            key=lambda row: row["eval_order"],
        )
        by_eval = {row["eval_id"]: row for row in group}
        nominal = by_eval[FAMILIES[family]["nominal"]]
        hardest = by_eval[FAMILIES[family]["hardest"]]
        summaries.append({
            "family": family,
            "train_id": train_id,
            "gap_train": group[0]["gap_train"],
            "dr": group[0]["dr"],
            "nominal_reward": nominal["mean_step_reward"],
            "nominal_episode_length": nominal["mean_episode_length"],
            "hardest_reward": hardest["mean_step_reward"],
            "hardest_episode_length": hardest["mean_episode_length"],
            "hardest_horizon_survival": hardest["horizon_survival"],
            "hardest_terminations": hardest["non_timeout_terminations"],
            "hardest_error_vel_xy": hardest["error_vel_xy"],
            "hardest_error_vel_yaw": hardest["error_vel_yaw"],
            "mean_reward": mean(row["mean_step_reward"] for row in group),
            "mean_episode_length": mean(row["mean_episode_length"] for row in group),
            "mean_horizon_survival": mean(row["horizon_survival"] for row in group),
            "mean_error_vel_xy": mean(row["error_vel_xy"] for row in group),
            "mean_error_vel_yaw": mean(row["error_vel_yaw"] for row in group),
        })

    for family in FAMILIES:
        family_rows = [row for row in summaries if row["family"] == family]
        family_rows.sort(key=lambda row: (
            -row["hardest_horizon_survival"],
            -row["mean_horizon_survival"],
            -row["hardest_reward"],
            row["hardest_error_vel_xy"],
            row["hardest_error_vel_yaw"],
        ))
        for rank, row in enumerate(family_rows, start=1):
            row["robust_rank"] = rank
    return sorted(summaries, key=lambda row: (row["family"], row["robust_rank"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summaries: list[dict], row_count: int) -> None:
    lines = [
        "# Go2 Gap Expert Cross-Evaluation",
        "",
        f"Completed evaluation cells: **{row_count}/42**.",
        "",
        "Ranking is lexicographic: hardest-condition horizon survival, mean horizon survival, "
        "hardest-condition step reward, then velocity tracking error. Payload and RR-calf "
        "families are ranked separately.",
        "",
    ]
    for family, title in (("payload", "Payload 0-10 kg"), ("rr_calf", "RR calf strength 1.0-0.5")):
        lines.extend([
            f"## {title}",
            "",
            "| Rank | Policy | Hard episode | Hard survival | Hard reward | Hard xy error | Mean reward | Nominal episode |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in (item for item in summaries if item["family"] == family):
            lines.append(
                f"| {row['robust_rank']} | {row['train_id']} | "
                f"{row['hardest_episode_length']:.1f} | {row['hardest_horizon_survival']:.3f} | "
                f"{row['hardest_reward']:.5f} | {row['hardest_error_vel_xy']:.4f} | "
                f"{row['mean_reward']:.5f} | {row['nominal_episode_length']:.1f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_family("payload", args.payload_root, args.steps)
    rows += read_family("rr_calf", args.rr_root, args.steps)
    counts = {family: sum(row["family"] == family for row in rows) for family in FAMILIES}
    missing = {
        family: config["expected"] - counts[family]
        for family, config in FAMILIES.items()
        if counts[family] != config["expected"]
    }
    if missing and not args.allow_incomplete:
        raise SystemExit(f"Incomplete cross-evaluation: counts={counts}, missing={missing}")
    if missing:
        print(f"warning: incomplete cross-evaluation counts={counts}")
        return

    summaries = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["family"], row["train_id"], row["eval_order"]))
    write_csv(args.output_dir / "all_metrics.csv", rows)
    write_csv(args.output_dir / "policy_summary.csv", summaries)
    write_report(args.output_dir / "evaluation_report.md", summaries, len(rows))
    print(f"wrote {args.output_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
