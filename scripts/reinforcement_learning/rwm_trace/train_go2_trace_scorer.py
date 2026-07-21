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
from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path


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
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--required_val_modes", nargs="*", default=[])
    parser.add_argument("--min_val_mode_count", type=int, default=0)
    parser.add_argument("--split_search_trials", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_no_validation", action="store_true")
    parser.add_argument(
        "--zero_all_missing_feature_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Zero first-layer value and missingness columns for features absent from every "
            "training summary. This prevents untrained random weights in an initial real-data scorer."
        ),
    )
    return parser.parse_args()


def _zero_all_missing_input_weights(
    model: Go2TraceScorer,
    raw_features: np.ndarray,
) -> list[str]:
    feature_count = len(GO2_FEATURE_NAMES)
    if raw_features.ndim != 2 or raw_features.shape[1] != 2 * feature_count:
        raise ValueError(
            f"Expected raw scorer features [N, {2 * feature_count}], got {raw_features.shape}."
        )
    all_missing = np.all(raw_features[:, feature_count:] > 0.5, axis=0)
    missing_indices = np.flatnonzero(all_missing).tolist()
    zero_columns = missing_indices + [feature_count + index for index in missing_indices]
    with torch.no_grad():
        if zero_columns:
            model.net[0].weight[:, zero_columns] = 0.0
    return [GO2_FEATURE_NAMES[index] for index in missing_indices]


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


def _start_key(summary: dict) -> str:
    return str(summary.get("start_state_key", summary.get("start_state_id", -1)))


def _comparison_group_key(summary: dict) -> str:
    """Return the indivisible train/validation group for one trajectory.

    Dataset windows from one episode share a group even when their exact start
    states differ. Simulator counterfactual branches normally share both keys.
    """

    return str(summary.get("comparison_group_key", _start_key(summary)))


def _component_group_keys(summaries: list[dict], pair_count: int) -> list[str]:
    """Group pairs by connected components of their comparison-group graph.

    Splitting by the pair tuple leaks a start state whenever cross-start pairs
    connect it to more than one partner.  Connected components are the smallest
    units that guarantee no episode, start state, or trajectory crosses the
    train/validation boundary.
    """

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    starts: list[tuple[str, str]] = []
    for index in range(pair_count):
        left = _comparison_group_key(summaries[2 * index])
        right = _comparison_group_key(summaries[2 * index + 1])
        starts.append((left, right))
        union(left, right)
    return [find(left) for left, _ in starts]


def _balanced_accuracy(prediction: torch.Tensor, label: torch.Tensor) -> float:
    values = []
    for target in (False, True):
        mask = label.bool() == target
        if not bool(mask.any()):
            return float("nan")
        values.append(float((prediction[mask] == label.bool()[mask]).float().mean()))
    return float(np.mean(values)) if values else float("nan")


def _per_mode_metrics(
    pair_indices: list[int],
    summaries: list[dict],
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    modes = sorted(
        {
            str(summary.get("command_mode", "unknown"))
            for summary in summaries
        }
    )
    result: dict[str, dict[str, float | int]] = {}
    for mode in modes:
        indices = [
            index
            for index in pair_indices
            if str(summaries[2 * index].get("command_mode", "unknown")) == mode
            and str(summaries[2 * index + 1].get("command_mode", "unknown")) == mode
        ]
        if not indices:
            result[mode] = {"count": 0, "accuracy": float("nan"), "balanced_accuracy": float("nan")}
            continue
        index_t = torch.tensor(indices, dtype=torch.long)
        result[mode] = {
            "count": len(indices),
            "count_i": int(labels[index_t].bool().sum()),
            "count_j": int((~labels[index_t].bool()).sum()),
            "accuracy": float((predictions[index_t] == labels[index_t].bool()).float().mean()),
            "balanced_accuracy": _balanced_accuracy(predictions[index_t], labels[index_t]),
        }
    return result


def _explicit_partition_indices(
    rows: list[dict],
    summaries: list[dict],
    required_modes: tuple[str, ...],
    min_mode_count: int,
) -> tuple[list[int], list[int]] | None:
    """Use a partition fixed before pairing and verify it is leakage-safe."""

    partitions = [row.get("split_partition") for row in rows]
    if not any(value is not None for value in partitions):
        return None
    if any(value not in {"train", "val"} for value in partitions):
        raise ValueError(
            "Explicit scorer split is partially missing or invalid; every labeled pair "
            "must have split_partition=train|val."
        )

    group_partition: dict[str, str] = {}
    for pair_index, partition in enumerate(partitions):
        assert isinstance(partition, str)
        for side in (0, 1):
            group = _comparison_group_key(summaries[2 * pair_index + side])
            previous = group_partition.setdefault(group, partition)
            if previous != partition:
                raise ValueError(
                    f"Explicit scorer split leakage: source group {group} appears in "
                    f"both {previous} and {partition}."
                )

    train_indices = [index for index, value in enumerate(partitions) if value == "train"]
    val_indices = [index for index, value in enumerate(partitions) if value == "val"]
    if not train_indices or not val_indices:
        raise ValueError("Explicit scorer split must contain both train and validation pairs.")

    for partition, indices in (("train", train_indices), ("val", val_indices)):
        stats = {mode: {"count": 0, "i": 0, "j": 0} for mode in required_modes}
        for index in indices:
            left_mode = str(summaries[2 * index].get("command_mode", "unknown"))
            right_mode = str(summaries[2 * index + 1].get("command_mode", "unknown"))
            mode = left_mode if left_mode == right_mode else "mixed"
            if mode not in stats:
                continue
            feedback = str(rows[index].get("feedback", ""))
            stats[mode]["count"] += 1
            if feedback in {"i", "j"}:
                stats[mode][feedback] += 1
        missing = {
            mode: values
            for mode, values in stats.items()
            if values["count"] < min_mode_count or values["i"] < 1 or values["j"] < 1
        }
        if missing:
            raise ValueError(
                f"Explicit {partition} scorer partition lacks required per-mode label coverage: "
                f"required_count={min_mode_count}, required_i_j>=1, available={missing}"
            )
    return train_indices, val_indices


def _select_validation_groups(
    group_keys: list[str],
    pair_modes: list[str],
    pair_labels: list[bool],
    val_group_count: int,
    rng: np.random.Generator,
    required_modes: tuple[str, ...],
    min_mode_count: int,
    search_trials: int,
) -> set[str]:
    """Choose leak-free validation groups with per-mode train/val label coverage."""

    unique_groups = sorted(set(group_keys))
    if val_group_count <= 0:
        return set()
    if val_group_count >= len(unique_groups):
        raise ValueError("Validation split would consume every connected component.")
    if not required_modes:
        return set(rng.choice(unique_groups, size=val_group_count, replace=False).tolist())

    def empty_stats() -> dict[str, dict[str, int]]:
        return {mode: {"count": 0, "i": 0, "j": 0} for mode in required_modes}

    def add_one(stats: dict[str, dict[str, int]], mode: str, label_i: bool, delta: int = 1) -> None:
        if mode not in stats:
            return
        row = stats[mode]
        row["count"] += delta
        row["i" if label_i else "j"] += delta

    group_stats: dict[str, dict[str, dict[str, int]]] = {}
    total_stats = empty_stats()
    for group, mode, label_i in zip(group_keys, pair_modes, pair_labels, strict=True):
        stats = group_stats.setdefault(group, empty_stats())
        add_one(stats, mode, label_i)
        add_one(total_stats, mode, label_i)

    impossible = {}
    for mode in required_modes:
        row = total_stats[mode]
        if row["count"] < 2 * min_mode_count or row["i"] < 2 or row["j"] < 2:
            impossible[mode] = dict(row)
    if impossible:
        raise ValueError(
            "Required command-mode train/validation coverage is absent from labeled data: "
            f"required_count_each_split={min_mode_count}, required_i_j_each_split>=1, available={impossible}"
        )

    def combine(groups: set[str]) -> dict[str, dict[str, int]]:
        out = empty_stats()
        for group in groups:
            for mode, row in group_stats[group].items():
                out[mode]["count"] += row["count"]
                out[mode]["i"] += row["i"]
                out[mode]["j"] += row["j"]
        return out

    def subtract(total: dict[str, dict[str, int]], part: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        return {
            mode: {key: total[mode][key] - part[mode][key] for key in ("count", "i", "j")}
            for mode in required_modes
        }

    def deficits(stats: dict[str, dict[str, int]]) -> int:
        value = 0
        for mode in required_modes:
            row = stats[mode]
            value += max(0, min_mode_count - row["count"])
            value += max(0, 1 - row["i"])
            value += max(0, 1 - row["j"])
        return value

    def total_deficits(groups: set[str]) -> int:
        val_stats = combine(groups)
        train_stats = subtract(total_stats, val_stats)
        return deficits(val_stats) + deficits(train_stats)

    def can_add(groups: set[str], group: str) -> bool:
        candidate = set(groups)
        candidate.add(group)
        train_stats = subtract(total_stats, combine(candidate))
        return deficits(train_stats) == 0

    # Construct the required validation coverage first. This avoids relying on
    # random search for rare command modes.
    constructed: set[str] = set()
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)
    for mode in required_modes:
        while True:
            val_stats = combine(constructed)
            row = val_stats[mode]
            if row["count"] >= min_mode_count and row["i"] >= 1 and row["j"] >= 1:
                break
            candidates = [
                group for group in shuffled_groups
                if group not in constructed
                and group_stats[group][mode]["count"] > 0
                and can_add(constructed, group)
            ]
            if not candidates:
                break
            def score(group: str) -> tuple[int, int, int]:
                before = deficits({mode: dict(row) for mode, row in val_stats.items()})
                trial = set(constructed); trial.add(group)
                after = deficits(combine(trial))
                contribution = group_stats[group][mode]
                label_bonus = int(row["i"] == 0 and contribution["i"] > 0) + int(row["j"] == 0 and contribution["j"] > 0)
                return (before - after, label_bonus, contribution["count"])
            constructed.add(max(candidates, key=score))

    if total_deficits(constructed) == 0:
        while len(constructed) < val_group_count:
            candidates = [group for group in shuffled_groups if group not in constructed and can_add(constructed, group)]
            if not candidates:
                break
            constructed.add(candidates[0])
        if total_deficits(constructed) == 0:
            return constructed

    best_groups: set[str] | None = constructed if constructed else None
    best_score: tuple[int, int, int, int] | None = None
    trials = max(1, search_trials)
    for _ in range(trials):
        candidate = set(rng.choice(unique_groups, size=val_group_count, replace=False).tolist())
        val_stats = combine(candidate)
        train_stats = subtract(total_stats, val_stats)
        td = deficits(val_stats) + deficits(train_stats)
        val_minimum = min((row["count"] for row in val_stats.values()), default=0)
        train_minimum = min((row["count"] for row in train_stats.values()), default=0)
        score = (-td, val_minimum, train_minimum, sum(row["count"] for row in val_stats.values()))
        if best_score is None or score > best_score:
            best_score = score
            best_groups = candidate
        if td == 0:
            return candidate

    assert best_groups is not None
    val_stats = combine(best_groups)
    train_stats = subtract(total_stats, val_stats)
    raise ValueError(
        "Could not construct a leak-free validation split with required command and label coverage: "
        f"required_count_each_split={min_mode_count}, val_stats={val_stats}, "
        f"train_stats={train_stats}, val_groups={val_group_count}, trials={trials}"
    )


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
    rng = np.random.default_rng(args.seed)
    pair_modes = [
        str(summaries[2 * index].get("command_mode", "unknown"))
        if str(summaries[2 * index].get("command_mode", "unknown"))
        == str(summaries[2 * index + 1].get("command_mode", "unknown"))
        else "mixed"
        for index in range(len(rows))
    ]
    pair_label_bools = [row["feedback"] == "i" for row in rows]
    explicit_split = _explicit_partition_indices(
        rows,
        summaries,
        tuple(args.required_val_modes),
        args.min_val_mode_count,
    )
    if explicit_split is not None:
        train_indices, val_indices = explicit_split
        split_strategy = "preassigned_before_pairing"
    else:
        group_keys = _component_group_keys(summaries, len(rows))
        unique_groups = sorted(set(group_keys))
        if len(unique_groups) > 1:
            val_group_count = int(round(len(unique_groups) * args.val_ratio))
            if args.required_val_modes:
                val_group_count = max(val_group_count, len(args.required_val_modes))
            val_group_count = min(val_group_count, len(unique_groups) - 1)
        else:
            val_group_count = 0
        val_groups = _select_validation_groups(
            group_keys,
            pair_modes,
            pair_label_bools,
            val_group_count,
            rng,
            tuple(args.required_val_modes),
            args.min_val_mode_count,
            args.split_search_trials,
        )
        train_indices = [index for index, key in enumerate(group_keys) if key not in val_groups]
        val_indices = [index for index, key in enumerate(group_keys) if key in val_groups]
        split_strategy = "connected_component_search"
    if not train_indices:
        raise ValueError("Connected-component split produced no training pairs.")
    if not val_indices and not args.allow_no_validation:
        raise ValueError(
            "Connected-component split produced no validation pairs. Use more independent "
            "start states or disable cross-start pairs; do not report training accuracy as validation."
        )

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
    confidence_weights = torch.tensor(
        [max(float(row.get("confidence", 1.0)), 1.0e-6) for row in rows],
        dtype=torch.float32,
    )
    train_index_t = torch.tensor(train_indices, dtype=torch.long)
    val_index_t = torch.tensor(val_indices, dtype=torch.long)

    model = Go2TraceScorer(features_t.shape[-1], args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    final_loss = float("nan")
    for _ in range(args.epochs):
        logits = model(x_i[train_index_t]) - model(x_j[train_index_t])
        pair_losses = F.binary_cross_entropy_with_logits(
            logits,
            labels[train_index_t],
            reduction="none",
        )
        train_weights = confidence_weights[train_index_t]
        loss = torch.sum(pair_losses * train_weights) / torch.clamp_min(train_weights.sum(), 1.0e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    all_missing_feature_names: list[str] = []
    if args.zero_all_missing_feature_weights:
        train_raw = features[train_feature_indices]
        all_missing_feature_names = _zero_all_missing_input_weights(model, train_raw)
    with torch.no_grad():
        predictions = model(x_i) > model(x_j)
        train_accuracy = float((predictions[train_index_t] == labels[train_index_t].bool()).float().mean())
        val_accuracy = (
            float((predictions[val_index_t] == labels[val_index_t].bool()).float().mean())
            if val_indices
            else float("nan")
        )
        train_balanced_accuracy = _balanced_accuracy(predictions[train_index_t], labels[train_index_t])
        val_balanced_accuracy = (
            _balanced_accuracy(predictions[val_index_t], labels[val_index_t])
            if val_indices
            else float("nan")
        )
        swapped_predictions = model(x_j) < model(x_i)
        pair_swap_consistency = float((predictions == swapped_predictions).float().mean())
        train_per_mode = _per_mode_metrics(train_indices, summaries, predictions, labels)
        val_per_mode = _per_mode_metrics(val_indices, summaries, predictions, labels)

    train_start_ids = {
        _start_key(summaries[2 * index + side])
        for index in train_indices
        for side in (0, 1)
    }
    val_start_ids = {
        _start_key(summaries[2 * index + side])
        for index in val_indices
        for side in (0, 1)
    }
    start_overlap = sorted(train_start_ids & val_start_ids)
    if start_overlap:
        raise RuntimeError(f"Scorer split leakage detected for start IDs: {start_overlap[:20]}")
    train_groups = {
        _comparison_group_key(summaries[2 * index + side])
        for index in train_indices
        for side in (0, 1)
    }
    val_groups_exact = {
        _comparison_group_key(summaries[2 * index + side])
        for index in val_indices
        for side in (0, 1)
    }
    group_overlap = sorted(train_groups & val_groups_exact)
    if group_overlap:
        raise RuntimeError(f"Scorer split leakage detected for source groups: {group_overlap[:20]}")
    conditions = sorted({str(row.get("condition", "general")) for row in rows})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_scorer_checkpoint(
        str(output),
        model,
        stats,
        hidden_dim=args.hidden_dim,
        metadata={
            "labels": str(Path(args.labels).resolve()),
            "labels_sha256": sha256_path(args.labels),
            "pairs": str(Path(args.pairs).resolve()) if args.pairs else None,
            "pairs_sha256": sha256_path(args.pairs) if args.pairs else None,
            "num_pairs": len(rows),
            "num_train_pairs": len(train_indices),
            "num_val_pairs": len(val_indices),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_balanced_accuracy": train_balanced_accuracy,
            "val_balanced_accuracy": val_balanced_accuracy,
            "pair_swap_consistency": pair_swap_consistency,
            "train_val_start_overlap": 0,
            "train_val_group_overlap": 0,
            "split_strategy": split_strategy,
            "conditions": conditions,
            "confidence_weighted_loss": True,
            "zero_all_missing_feature_weights": bool(args.zero_all_missing_feature_weights),
            "zeroed_all_missing_feature_names": all_missing_feature_names,
            "mean_pair_confidence": float(confidence_weights.mean()),
            "train_per_command_mode": train_per_mode,
            "val_per_command_mode": val_per_mode,
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
                "train_balanced_accuracy": train_balanced_accuracy,
                "val_balanced_accuracy": val_balanced_accuracy,
                "pair_swap_consistency": pair_swap_consistency,
                "train_val_start_overlap": 0,
                "train_per_command_mode": train_per_mode,
                "val_per_command_mode": val_per_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
