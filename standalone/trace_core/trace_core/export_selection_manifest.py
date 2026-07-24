#!/usr/bin/env python3
"""Freeze selected row IDs independently of the rule/scorer implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import (
    IDENTITY_FIELDS,
    atomic_write_json,
    identity,
    read_jsonl,
    sha256_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--candidate-dataset", required=True)
    parser.add_argument("--selected-indices-json", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _score_source(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") == "portable_trace_rule_scores_v1":
        return dict(manifest["score_source"])
    if manifest.get("schema") == "portable_trace_scorer_scores_v1":
        scorer = dict(manifest["scorer"])
        return {
            "kind": "learned_scorer",
            "checkpoint_sha256": scorer["checkpoint_sha256"],
            "format_version": scorer["format_version"],
            "feature_schema": scorer["feature_schema"],
        }
    raise ValueError("Unsupported TRACE score manifest schema.")


def main() -> None:
    args = parse_args()
    summaries_path = Path(args.summaries).expanduser().resolve()
    scores_path = Path(args.scores).expanduser().resolve()
    score_manifest_path = Path(args.score_manifest).expanduser().resolve()
    candidate_path = Path(args.candidate_dataset).expanduser().resolve()
    selection_path = Path(args.selected_indices_json).expanduser().resolve()
    summaries = read_jsonl(summaries_path)
    scores = read_jsonl(scores_path)
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("summaries_sha256") != sha256_path(summaries_path):
        raise ValueError("Score manifest summary hash mismatch.")
    if score_manifest.get("scores_sha256") != sha256_path(scores_path):
        raise ValueError("Score manifest score-file hash mismatch.")
    if len(summaries) != len(scores):
        raise ValueError("Summary and score row counts differ.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_indices = [int(value) for value in selection["selected_indices"]]
    if (
        len(selected_indices) != len(set(selected_indices))
        or any(index < 0 or index >= len(summaries) for index in selected_indices)
    ):
        raise ValueError("External selector contains invalid or duplicate indices.")
    for index, score in enumerate(scores):
        if int(score["row_index"]) != index:
            raise ValueError(f"Score row index mismatch at {index}.")
        for field in IDENTITY_FIELDS:
            if field in score and score[field] != summaries[index].get(field):
                raise ValueError(
                    f"Score identity mismatch at row {index}, field {field}."
                )
    selected_identities = [
        {
            **identity(summaries[index], index),
            **{
                field: summaries[index].get(field)
                for field in IDENTITY_FIELDS
                if field in summaries[index]
            },
        }
        for index in selected_indices
    ]
    manifest = {
        "schema": "portable_trace_selection_v1",
        "candidate_dataset": str(candidate_path),
        "candidate_dataset_sha256": sha256_path(candidate_path),
        "summaries": str(summaries_path),
        "summaries_sha256": sha256_path(summaries_path),
        "score_manifest": str(score_manifest_path),
        "score_manifest_sha256": sha256_path(score_manifest_path),
        "score_source": _score_source(score_manifest),
        "row_count": len(summaries),
        "selected_count": len(selected_indices),
        "selected_indices": selected_indices,
        "selected_identities": selected_identities,
        "scores": [float(row["score"]) for row in scores],
        "selection_policy": str(selection_path),
        "selection_policy_sha256": sha256_path(selection_path),
    }
    atomic_write_json(args.output, manifest)
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
