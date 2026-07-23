#!/usr/bin/env python3
"""Export a portable selected-index contract from scored trajectory summaries."""

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

from trace_scorer.core import read_jsonl, sha256_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--candidate-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-field", default="selected")
    parser.add_argument(
        "--selected-indices-json",
        default=None,
        help="Optional external-selector JSON containing selected_indices.",
    )
    parser.add_argument("--reference-score-field", default="trace_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries_path = Path(args.summaries).expanduser().resolve()
    scores_path = Path(args.scores).expanduser().resolve()
    score_manifest_path = Path(args.score_manifest).expanduser().resolve()
    candidate_path = Path(args.candidate_dataset).expanduser().resolve()
    summaries = read_jsonl(summaries_path)
    score_rows = read_jsonl(scores_path)
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("schema") != "portable_trace_scorer_scores_v1":
        raise ValueError("Not a portable TRACE score manifest.")
    if score_manifest.get("summaries_sha256") != sha256_path(summaries_path):
        raise ValueError("Score manifest summary hash mismatch.")
    if score_manifest.get("scores_sha256") != sha256_path(scores_path):
        raise ValueError("Score manifest score-file hash mismatch.")
    if len(summaries) != len(score_rows):
        raise ValueError("Summary and score row counts differ.")
    external_indices: list[int] | None = None
    selection_policy = "summary_selected_field"
    selection_policy_sha256 = None
    if args.selected_indices_json:
        selection_path = Path(args.selected_indices_json).expanduser().resolve()
        selection_document = json.loads(selection_path.read_text(encoding="utf-8"))
        external_indices = [int(value) for value in selection_document["selected_indices"]]
        if (
            len(set(external_indices)) != len(external_indices)
            or any(index < 0 or index >= len(summaries) for index in external_indices)
        ):
            raise ValueError("External selector contains invalid or duplicate indices.")
        selection_policy = str(selection_path)
        selection_policy_sha256 = sha256_path(selection_path)
    selected_set = set(external_indices or [])
    scores = np.empty(len(summaries), dtype=np.float32)
    max_reference_error = 0.0
    selected_indices: list[int] = []
    selected_identities: list[dict] = []
    identity_fields = (
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
    for index, (summary, score_row) in enumerate(
        zip(summaries, score_rows, strict=True)
    ):
        if int(score_row["row_index"]) != index:
            raise ValueError(f"Score row index mismatch at {index}.")
        score = float(score_row["score"])
        scores[index] = score
        if args.reference_score_field in summary:
            max_reference_error = max(
                max_reference_error,
                abs(score - float(summary[args.reference_score_field])),
            )
        is_selected = (
            index in selected_set
            if external_indices is not None
            else bool(summary.get(args.selected_field, False))
        )
        if is_selected:
            selected_indices.append(index)
            selected_identities.append(
                {
                    "row_index": index,
                    **{
                        field: summary.get(field)
                        for field in identity_fields
                        if field in summary
                    },
                }
            )
    manifest = {
        "schema": "portable_trace_selection_v1",
        "candidate_dataset": str(candidate_path),
        "candidate_dataset_sha256": sha256_path(candidate_path),
        "summaries": str(summaries_path),
        "summaries_sha256": sha256_path(summaries_path),
        "score_manifest": str(score_manifest_path),
        "score_manifest_sha256": sha256_path(score_manifest_path),
        "scorer_checkpoint_sha256": score_manifest["scorer"]["checkpoint_sha256"],
        "row_count": len(summaries),
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
        "selected_identities": selected_identities,
        "scores": [float(value) for value in scores],
        "reference_score_field": args.reference_score_field,
        "max_reference_score_abs_error": max_reference_error,
        "selection_policy": selection_policy,
        "selection_policy_sha256": selection_policy_sha256,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"scores", "selected_indices", "selected_identities"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
