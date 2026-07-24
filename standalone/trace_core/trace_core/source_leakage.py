#!/usr/bin/env python3
"""Audit whether critic observations reveal RWM-versus-TRACE replay source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json, sha256_path


def _load_observations(path: str | Path, key: str) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if source.suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    else:
        with np.load(source, allow_pickle=False) as payload:
            if key not in payload:
                raise ValueError(f"{source} has no array named {key!r}.")
            values = payload[key]
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"Expected a non-empty [rows, features] array: {source}")
    if not np.isfinite(values).all():
        raise ValueError(f"Observation audit input contains non-finite values: {source}")
    return values


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both replay sources.")
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def audit_source_leakage(
    primary: np.ndarray,
    trace: np.ndarray,
    *,
    seed: int = 42,
    maximum_rows_per_source: int = 100_000,
    maximum_auc: float = 0.70,
    highlighted_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    primary = np.asarray(primary, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    if primary.ndim != 2 or trace.ndim != 2:
        raise ValueError("Replay observations must be rank-two arrays.")
    if primary.shape[1] != trace.shape[1]:
        raise ValueError("Primary and TRACE observation widths differ.")
    if not np.isfinite(primary).all() or not np.isfinite(trace).all():
        raise ValueError("Replay observations contain non-finite values.")
    if not 0.5 <= maximum_auc <= 1.0:
        raise ValueError("maximum_auc must be in [0.5, 1].")
    rng = np.random.default_rng(seed)
    count = min(
        len(primary),
        len(trace),
        int(maximum_rows_per_source),
    )
    if count < 100:
        raise ValueError("Source leakage audit requires at least 100 rows per source.")
    primary = primary[rng.choice(len(primary), count, replace=False)]
    trace = trace[rng.choice(len(trace), count, replace=False)]
    split = max(1, min(count - 1, int(round(0.70 * count))))
    primary = primary[rng.permutation(count)]
    trace = trace[rng.permutation(count)]
    train_x = np.concatenate([primary[:split], trace[:split]], axis=0)
    train_y = np.concatenate(
        [np.zeros(split), np.ones(split)],
        axis=0,
    )
    test_x = np.concatenate([primary[split:], trace[split:]], axis=0)
    test_y = np.concatenate(
        [np.zeros(count - split), np.ones(count - split)],
        axis=0,
    )
    train_order = rng.permutation(len(train_x))
    test_order = rng.permutation(len(test_x))
    train_x, train_y = train_x[train_order], train_y[train_order]
    test_x, test_y = test_x[test_order], test_y[test_order]
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    train_z = (train_x - mean) / std
    test_z = (test_x - mean) / std
    weights = np.zeros(train_z.shape[1], dtype=np.float64)
    bias = 0.0
    for step in range(300):
        probability = _sigmoid(train_z @ weights + bias)
        residual = probability - train_y
        learning_rate = 0.15 / np.sqrt(1.0 + step / 100.0)
        weights -= learning_rate * (
            train_z.T @ residual / len(train_z) + 1.0e-3 * weights
        )
        bias -= learning_rate * float(residual.mean())
    probabilities = _sigmoid(test_z @ weights + bias)
    raw_auc = float(_auc(test_y, probabilities))
    separation_auc = max(raw_auc, 1.0 - raw_auc)
    prediction = probabilities >= 0.5
    true_positive_rate = float(prediction[test_y == 1].mean())
    true_negative_rate = float((~prediction[test_y == 0]).mean())
    balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
    pooled_std = np.sqrt(
        0.5 * (primary.var(axis=0) + trace.var(axis=0))
    )
    standardized_mean_difference = np.divide(
        trace.mean(axis=0) - primary.mean(axis=0),
        pooled_std,
        out=np.zeros(primary.shape[1], dtype=np.float64),
        where=pooled_std > 1.0e-8,
    )
    ranked = np.argsort(-np.abs(standardized_mean_difference))
    highlighted = {
        str(index): {
            "primary_mean": float(primary[:, index].mean()),
            "trace_mean": float(trace[:, index].mean()),
            "standardized_mean_difference": float(
                standardized_mean_difference[index]
            ),
        }
        for index in highlighted_indices
        if 0 <= index < primary.shape[1]
    }
    return {
        "schema": "portable_trace_source_leakage_v1",
        "rows_per_source": count,
        "feature_count": int(primary.shape[1]),
        "linear_source_auc": separation_auc,
        "linear_source_balanced_accuracy": balanced_accuracy,
        "maximum_allowed_auc": float(maximum_auc),
        "passed": bool(separation_auc <= maximum_auc),
        "maximum_absolute_standardized_mean_difference": float(
            np.abs(standardized_mean_difference).max()
        ),
        "largest_shift_features": [
            {
                "index": int(index),
                "standardized_mean_difference": float(
                    standardized_mean_difference[index]
                ),
            }
            for index in ranked[: min(10, len(ranked))]
        ],
        "highlighted_features": highlighted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--array-key", default="observation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximum-rows-per-source", type=int, default=100_000)
    parser.add_argument("--maximum-auc", type=float, default=0.70)
    parser.add_argument("--highlight-indices", default="0,1,2")
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = _load_observations(args.primary, args.array_key)
    trace = _load_observations(args.trace, args.array_key)
    highlighted = tuple(
        int(value)
        for value in args.highlight_indices.split(",")
        if value.strip()
    )
    report = audit_source_leakage(
        primary,
        trace,
        seed=args.seed,
        maximum_rows_per_source=args.maximum_rows_per_source,
        maximum_auc=args.maximum_auc,
        highlighted_indices=highlighted,
    )
    report["primary_path"] = str(Path(args.primary).expanduser().resolve())
    report["primary_sha256"] = sha256_path(args.primary)
    report["trace_path"] = str(Path(args.trace).expanduser().resolve())
    report["trace_sha256"] = sha256_path(args.trace)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.enforce and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
