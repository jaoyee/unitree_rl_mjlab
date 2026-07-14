#!/usr/bin/env python3
"""Summarize the aligned Go2 final-policy gap cross-evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


POLICIES = ("g0", "rr03", "rr05", "p5", "p75")
GAPS = ("g0", "rr03", "rr05", "p5", "p75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    return parser.parse_args()


def read_rows(root: Path) -> list[dict]:
    rows = []
    missing = []
    for policy in POLICIES:
        for gap in GAPS:
            path = root / policy / f"{gap}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "policy": policy,
                "gap": gap,
                "terminations": int(data["non_timeout_termination_count"]),
                "completed_episodes": int(data["completed_episodes"]),
                "mean_return": float(data["mean_return"]),
                "mean_episode_length": float(data["mean_episode_length"]),
                "error_vel_xy": float(data["error_vel_xy"]),
                "error_vel_yaw": float(data["error_vel_yaw"]),
                "action_abs_mean": float(data["action_abs_mean"]),
                "base_speed_xy": float(data["base_speed_xy"]),
                "num_envs": int(data["num_envs"]),
                "steps": int(data["steps"]),
                "metrics_path": str(path.resolve()),
            })
    if missing:
        raise SystemExit("Missing evaluation cells:\n" + "\n".join(missing))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        by_gap = {row["gap"]: row for row in group}
        matched = by_gap[policy]
        total_terminations = sum(row["terminations"] for row in group)
        summaries.append({
            "policy": policy,
            "total_terminations": total_terminations,
            "terminations_per_initial_env": total_terminations / group[0]["num_envs"],
            "zero_termination_gaps": sum(row["terminations"] == 0 for row in group),
            "worst_gap_terminations": max(row["terminations"] for row in group),
            "nominal_terminations": by_gap["g0"]["terminations"],
            "matched_terminations": matched["terminations"],
            "matched_error_vel_xy": matched["error_vel_xy"],
            "matched_error_vel_yaw": matched["error_vel_yaw"],
            "mean_error_vel_xy": mean(row["error_vel_xy"] for row in group),
            "mean_error_vel_yaw": mean(row["error_vel_yaw"] for row in group),
            "mean_action_abs": mean(row["action_abs_mean"] for row in group),
        })

    summaries.sort(key=lambda row: (
        row["total_terminations"],
        -row["zero_termination_gaps"],
        row["worst_gap_terminations"],
        row["mean_error_vel_xy"] + row["mean_error_vel_yaw"],
    ))
    for rank, row in enumerate(summaries, start=1):
        row["robust_rank"] = rank
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], summaries: list[dict]) -> None:
    lines = [
        "# Aligned Go2 Policy Gap Cross-Evaluation",
        "",
        "All policies use the same 8-command schedule, 2400 steps, 256 environments, "
        "seed 400, and clean physics apart from the explicitly injected gap.",
        "",
        "Ranking is lexicographic: total non-timeout terminations, number of zero-termination "
        "gap cells, worst-cell terminations, then mean velocity-tracking error. Mean return is "
        "reported per cell but is not used for ranking because auto-reset makes completed-episode "
        "return incomparable when only a subset of environments terminates.",
        "",
        "## Robustness Ranking",
        "",
        "| Rank | Policy | Total term. | Zero-term gaps | Worst term. | Nominal term. | Matched term. | Mean xy err. | Mean yaw err. |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['robust_rank']} | {row['policy']} | {row['total_terminations']} | "
            f"{row['zero_termination_gaps']}/5 | {row['worst_gap_terminations']} | "
            f"{row['nominal_terminations']} | {row['matched_terminations']} | "
            f"{row['mean_error_vel_xy']:.4f} | {row['mean_error_vel_yaw']:.4f} |"
        )

    lines.extend([
        "",
        "## Termination Matrix",
        "",
        "Rows are policies and columns are evaluation gaps.",
        "",
        "| Policy | g0 | RR0.3 | RR0.5 | 5 kg | 7.5 kg |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_cell = {(row["policy"], row["gap"]): row for row in rows}
    for policy in POLICIES:
        values = [str(by_cell[(policy, gap)]["terminations"]) for gap in GAPS]
        lines.append(f"| {policy} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Velocity Error Matrix",
        "",
        "Each cell is `xy / yaw` mean absolute tracking error.",
        "",
        "| Policy | g0 | RR0.3 | RR0.5 | 5 kg | 7.5 kg |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for policy in POLICIES:
        values = [
            f"{by_cell[(policy, gap)]['error_vel_xy']:.3f} / "
            f"{by_cell[(policy, gap)]['error_vel_yaw']:.3f}"
            for gap in GAPS
        ]
        lines.append(f"| {policy} | " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.eval_root)
    summaries = summarize(rows)
    write_csv(args.eval_root / "all_metrics.csv", rows)
    write_csv(args.eval_root / "policy_summary.csv", summaries)
    write_report(args.eval_root / "evaluation_report.md", rows, summaries)
    (args.eval_root / "summary.json").write_text(
        json.dumps({"rows": rows, "policies": summaries}, indent=2),
        encoding="utf-8",
    )
    print(args.eval_root / "evaluation_report.md")


if __name__ == "__main__":
    main()
