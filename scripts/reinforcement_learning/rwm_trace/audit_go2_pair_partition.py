#!/usr/bin/env python3
"""Fail-closed audit for partitioned Go2 feedback pairs and optional labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required_modes", nargs="+", required=True)
    parser.add_argument("--min_pairs_per_partition_mode", type=int, default=1)
    parser.add_argument("--min_labels_per_partition_mode", type=int, default=0)
    parser.add_argument("--min_labels_per_side", type=int, default=0)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _group(summary: dict) -> str:
    return str(summary.get("comparison_group_key", summary.get("start_state_key", summary.get("start_state_id", -1))))


def _mode(pair: dict) -> str:
    left = str(pair.get("trajectory_i", {}).get("command_mode", "unknown"))
    right = str(pair.get("trajectory_j", {}).get("command_mode", "unknown"))
    return left if left == right else "mixed"


def main() -> None:
    args = parse_args()
    required_modes = tuple(args.required_modes)
    pairs = _read_jsonl(args.pairs)
    if not pairs:
        raise ValueError("Pair file is empty.")
    pair_ids: set[str] = set()
    pair_by_id: dict[str, dict] = {}
    group_partition: dict[str, str] = {}
    pair_stats = {partition: {mode: 0 for mode in required_modes} for partition in ("train", "val")}
    errors: list[str] = []
    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id or pair_id in pair_ids:
            errors.append(f"missing_or_duplicate_pair_id:{pair_id}")
            continue
        pair_ids.add(pair_id)
        pair_by_id[pair_id] = pair
        partition = str(pair.get("split_partition", ""))
        if partition not in {"train", "val"}:
            errors.append(f"bad_partition:{pair_id}:{partition}")
            continue
        mode = _mode(pair)
        if mode not in required_modes:
            errors.append(f"bad_mode:{pair_id}:{mode}")
            continue
        pair_stats[partition][mode] += 1
        for side in ("trajectory_i", "trajectory_j"):
            group = _group(pair.get(side, {}))
            previous = group_partition.setdefault(group, partition)
            if previous != partition:
                errors.append(f"group_leakage:{group}:{previous}:{partition}")

    pair_shortfalls = {
        f"{partition}/{mode}": count
        for partition, rows in pair_stats.items()
        for mode, count in rows.items()
        if count < args.min_pairs_per_partition_mode
    }

    label_stats = {
        partition: {mode: {"count": 0, "i": 0, "j": 0} for mode in required_modes}
        for partition in ("train", "val")
    }
    unknown_label_ids: list[str] = []
    if args.labels:
        for label in _read_jsonl(args.labels):
            pair_id = str(label.get("pair_id", ""))
            pair = pair_by_id.get(pair_id)
            if pair is None:
                unknown_label_ids.append(pair_id)
                continue
            feedback = str(label.get("feedback", ""))
            if feedback not in {"i", "j"} or float(label.get("confidence", 0.0)) < args.confidence_threshold:
                continue
            partition = str(pair["split_partition"])
            mode = _mode(pair)
            row = label_stats[partition][mode]
            row["count"] += 1
            row[feedback] += 1

    label_shortfalls = {}
    if args.labels:
        for partition, rows in label_stats.items():
            for mode, row in rows.items():
                if (
                    row["count"] < args.min_labels_per_partition_mode
                    or row["i"] < args.min_labels_per_side
                    or row["j"] < args.min_labels_per_side
                ):
                    label_shortfalls[f"{partition}/{mode}"] = row

    report = {
        "protocol": "stable_group_partition_v1",
        "pair_count": len(pairs),
        "independent_group_count": len(group_partition),
        "pair_stats": pair_stats,
        "pair_shortfalls": pair_shortfalls,
        "label_stats": label_stats if args.labels else None,
        "label_shortfalls": label_shortfalls,
        "unknown_label_ids": unknown_label_ids[:20],
        "errors": errors[:50],
        "passed": not errors and not pair_shortfalls and not label_shortfalls and not unknown_label_ids,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".new")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
