#!/usr/bin/env python3
"""Select a scorer seed using one fixed, leakage-safe validation partition."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--required_modes", nargs="+", required=True)
    parser.add_argument("--min_global_balanced_accuracy", type=float, default=0.60)
    parser.add_argument("--min_mode_balanced_accuracy", type=float, default=0.55)
    parser.add_argument("--min_mode_count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    accepted = []
    for value in args.candidate:
        path = Path(value).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = dict(checkpoint.get("metadata") or {})
        global_accuracy = float(metadata.get("val_balanced_accuracy", float("nan")))
        mode_metrics = dict(metadata.get("val_per_command_mode") or {})
        failures = []
        mode_accuracies = []
        for mode in args.required_modes:
            metric = dict(mode_metrics.get(mode) or {})
            accuracy = float(metric.get("balanced_accuracy", float("nan")))
            mode_accuracies.append(accuracy)
            if (
                int(metric.get("count", 0)) < args.min_mode_count
                or int(metric.get("count_i", 0)) < 1
                or int(metric.get("count_j", 0)) < 1
                or not math.isfinite(accuracy)
                or accuracy < args.min_mode_balanced_accuracy
            ):
                failures.append({"mode": mode, "metrics": metric})
        if not math.isfinite(global_accuracy) or global_accuracy < args.min_global_balanced_accuracy:
            failures.append({"global_balanced_accuracy": global_accuracy})
        minimum_mode_accuracy = min(mode_accuracies) if mode_accuracies else float("nan")
        row = {
            "path": str(path),
            "global_balanced_accuracy": global_accuracy,
            "minimum_mode_balanced_accuracy": minimum_mode_accuracy,
            "failures": failures,
            "accepted": not failures,
        }
        rows.append(row)
        if not failures:
            accepted.append((minimum_mode_accuracy, global_accuracy, str(path), path))

    report = {"candidates": rows, "passed": bool(accepted)}
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".new")
    if not accepted:
        temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_report.replace(report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("No scorer seed passed the fixed validation quality gate.")

    accepted.sort(reverse=True)
    selected = accepted[0][3]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".new")
    shutil.copy2(selected, temporary_output)
    temporary_output.replace(output)
    report["selected"] = str(selected)
    report["output"] = str(output)
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
