#!/usr/bin/env python3
"""Train a Go2 TRACE scorer from confidence-filtered pairwise labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.scorer import (
    GO2_FEATURE_NAMES,
    Go2TraceScorer,
    feature_matrix,
    fit_feature_stats,
    normalize_features,
    save_scorer_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--pairs",
        default=None,
        help="Original pair JSONL when label rows only contain pair_id/feedback.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(row: dict, side: str) -> dict:
    embedded = row.get(f"trajectory_{side}") or row.get(f"traj_{side}_summary")
    if embedded is not None:
        return dict(embedded)
    path = row.get(f"traj_{side}_summary_path")
    if not path:
        raise ValueError(f"Label row lacks trajectory {side} summary.")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows = _read_jsonl(args.labels)
    if args.pairs:
        pairs = {str(row["pair_id"]): row for row in _read_jsonl(args.pairs)}
        rows = [{**pairs.get(str(row.get("pair_id")), {}), **row} for row in rows]
    rows = [
        row
        for row in rows
        if row.get("feedback") in {"i", "j"} and float(row.get("confidence", 0.0)) >= args.confidence_threshold
    ]
    if len(rows) < 2:
        raise ValueError("Need at least two confidence-filtered non-tie pair labels.")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be in [0, 1).")
    summaries = [item for row in rows for item in (_summary(row, "i"), _summary(row, "j"))]
    group_keys = [
        tuple(
            sorted(
                (
                    int(summaries[2 * index].get("start_state_id", -1)),
                    int(summaries[2 * index + 1].get("start_state_id", -1)),
                )
            )
        )
        for index in range(len(rows))
    ]
    unique_groups = sorted(set(group_keys))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)
    val_group_count = int(round(len(unique_groups) * args.val_ratio)) if len(unique_groups) > 1 else 0
    val_groups = set(unique_groups[:val_group_count])
    train_indices = [index for index, key in enumerate(group_keys) if key not in val_groups]
    val_indices = [index for index, key in enumerate(group_keys) if key in val_groups]
    if not train_indices:
        train_indices = list(range(len(rows)))
        val_indices = []

    features = feature_matrix(summaries, GO2_FEATURE_NAMES)
    train_feature_indices = [
        feature_index
        for pair_index in train_indices
        for feature_index in (2 * pair_index, 2 * pair_index + 1)
    ]
    stats = fit_feature_stats(features[train_feature_indices], GO2_FEATURE_NAMES)
    features_t = torch.from_numpy(normalize_features(features, stats))
    x_i = features_t[0::2]
    x_j = features_t[1::2]
    labels = torch.tensor([1.0 if row["feedback"] == "i" else 0.0 for row in rows])
    train_index_t = torch.tensor(train_indices, dtype=torch.long)
    val_index_t = torch.tensor(val_indices, dtype=torch.long)

    model = Go2TraceScorer(features_t.shape[-1], args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    final_loss = float("nan")
    for _ in range(args.epochs):
        logits = model(x_i[train_index_t]) - model(x_j[train_index_t])
        loss = F.binary_cross_entropy_with_logits(logits, labels[train_index_t])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    with torch.no_grad():
        predictions = model(x_i) > model(x_j)
        train_accuracy = float((predictions[train_index_t] == labels[train_index_t].bool()).float().mean())
        val_accuracy = (
            float((predictions[val_index_t] == labels[val_index_t].bool()).float().mean())
            if val_indices
            else float("nan")
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_scorer_checkpoint(
        str(output),
        model,
        stats,
        hidden_dim=args.hidden_dim,
        metadata={
            "labels": str(Path(args.labels).resolve()),
            "num_pairs": len(rows),
            "num_train_pairs": len(train_indices),
            "num_val_pairs": len(val_indices),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        },
    )
    print(
        json.dumps(
            {
                "num_pairs": len(rows),
                "num_train_pairs": len(train_indices),
                "num_val_pairs": len(val_indices),
                "final_loss": final_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
