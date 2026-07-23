#!/usr/bin/env python3
"""Score trajectory summaries without importing any RWM or replay implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from trace_scorer.core import load_scorer, read_jsonl, sha256_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def _identity(row: dict, index: int) -> dict:
    fields = (
        "candidate_namespace",
        "source_state_key",
        "comparison_group_key",
        "start_state_id",
        "trajectory_start",
        "trajectory_end",
        "command_mode",
        "command_vx_mean",
        "command_vy_mean",
        "command_yaw_mean",
    )
    return {"row_index": index, **{field: row.get(field) for field in fields if field in row}}


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summaries).expanduser().resolve()
    scorer_path = Path(args.scorer).expanduser().resolve()
    rows = read_jsonl(summary_path)
    scorer = load_scorer(scorer_path, device=args.device, reject_reward_features=True)
    scores, missing_counts = scorer.score(rows, batch_size=args.batch_size)
    scores_output = Path(args.scores_output).expanduser().resolve()
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_scores = scores_output.with_name(f"{scores_output.name}.tmp.{os.getpid()}")
    with temporary_scores.open("w", encoding="utf-8") as handle:
        for index, (row, score) in enumerate(zip(rows, scores, strict=True)):
            handle.write(
                json.dumps(
                    {
                        **_identity(row, index),
                        "score": float(score),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(temporary_scores, scores_output)
    present = {
        name: len(rows) - int(missing_counts[name])
        for name in scorer.base_feature_names
    }
    manifest = {
        "schema": "portable_trace_scorer_scores_v1",
        "summaries_path": str(summary_path),
        "summaries_sha256": sha256_path(summary_path),
        "scorer_path": str(scorer_path),
        "scorer": scorer.descriptor.to_dict(),
        "row_count": len(rows),
        "scores_path": str(scores_output),
        "scores_sha256": sha256_path(scores_output),
        "score_statistics": {
            "min": float(np.min(scores)) if len(scores) else None,
            "max": float(np.max(scores)) if len(scores) else None,
            "mean": float(np.mean(scores)) if len(scores) else None,
            "std": float(np.std(scores)) if len(scores) else None,
        },
        "feature_present_counts": present,
        "baseline_reward_features_rejected": True,
    }
    manifest_output = Path(args.manifest_output).expanduser().resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_output.with_name(
        f"{manifest_output.name}.tmp.{os.getpid()}"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
