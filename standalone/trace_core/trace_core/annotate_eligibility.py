#!/usr/bin/env python3
"""Attach the complete validated TRACE eligibility gate to trajectory summaries."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    validity = protocol["validity"]
    max_reset_error = float(validity["maximum_reset_reconstruction_error"])
    max_saturation = float(validity["maximum_saturation_fraction"])
    rows = [
        json.loads(line)
        for line in Path(args.summaries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: dict[str, int] = {}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            reason = None
            reset_error = float(row.get("reset_reconstruction_error", float("nan")))
            saturation = float(row.get("action_saturation_fraction", float("inf")))
            if bool(row.get("nonfinite_flag", True)):
                reason = "nonfinite"
            elif not math.isfinite(reset_error):
                reason = "reset_reconstruction_error_nonfinite"
            elif reset_error > max_reset_error:
                reason = "reset_reconstruction_error"
            elif saturation > max_saturation:
                reason = "action_saturation"
            elif not bool(row.get("command_motion_gate_passed", False)):
                reason = str(
                    row.get("command_motion_gate_rejection_reason")
                    or "command_motion_gate"
                )
            row["trace_selection_eligible"] = reason is None
            row["trace_selection_rejection_reason"] = reason
            if reason is not None:
                counts[reason] = counts.get(reason, 0) + 1
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "schema": "portable_trace_eligibility_v1",
                "row_count": len(rows),
                "eligible_count": len(rows) - sum(counts.values()),
                "rejection_counts": counts,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
