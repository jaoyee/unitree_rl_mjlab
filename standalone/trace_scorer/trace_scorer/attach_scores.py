#!/usr/bin/env python3
"""Attach portable scorer outputs to summaries for an external selector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from trace_scorer.core import read_jsonl, sha256_path


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-field", default="trace_score")
    args = parser.parse_args()
    summaries_path = Path(args.summaries).expanduser().resolve()
    scores_path = Path(args.scores).expanduser().resolve()
    manifest = json.loads(
        Path(args.score_manifest).expanduser().resolve().read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "portable_trace_scorer_scores_v1":
        raise ValueError("Not a portable TRACE score manifest.")
    if manifest.get("summaries_sha256") != sha256_path(summaries_path):
        raise ValueError("Score manifest summary hash mismatch.")
    if manifest.get("scores_sha256") != sha256_path(scores_path):
        raise ValueError("Score manifest score-file hash mismatch.")
    summaries = read_jsonl(summaries_path)
    scores = read_jsonl(scores_path)
    if len(summaries) != len(scores):
        raise ValueError("Summary and score row counts differ.")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for index, (summary, score) in enumerate(zip(summaries, scores, strict=True)):
            if int(score["row_index"]) != index:
                raise ValueError(f"Score row index mismatch at {index}.")
            row = dict(summary)
            row[args.score_field] = float(score["score"])
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "schema": "portable_trace_scored_summaries_v1",
                "row_count": len(summaries),
                "score_field": args.score_field,
                "output": str(output),
                "output_sha256": sha256_path(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
