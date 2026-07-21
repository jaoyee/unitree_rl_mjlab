#!/usr/bin/env python3
"""Build leakage-safe TRACE feedback pairs from a preassigned train/val split.

The split is assigned to stable source groups before any pair is created.  A
pair therefore cannot connect train and validation groups into one component,
which was the root cause of repeated initial-scorer repair failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.build_go2_feedback_pairs import (
    _command_mode,
    _comparison_group_key,
    _prompt,
    _read_jsonl,
    _start_key,
)
from scripts.reinforcement_learning.rwm_trace.reference_summary import (
    attach_references_to_summaries,
    load_reference_map,
)
from scripts.reinforcement_learning.rwm_trace.scorer import (
    GO2_FEATURE_NAMES,
    feature_matrix,
    load_scorer_checkpoint,
    score_summaries,
)

PARTITIONS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition_manifest", required=True)
    parser.add_argument("--scorer_checkpoint", default=None)
    parser.add_argument("--reference_summaries", default=None)
    parser.add_argument(
        "--condition",
        choices=("general", "g0", "rr05", "p5", "rr03", "p75"),
        default="general",
    )
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--pair_prefix", default="go2_pair")
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--required_pair_modes", nargs="*", default=[])
    parser.add_argument(
        "--min_pairs_per_mode_per_partition",
        type=int,
        default=0,
        help="Reserved pair count for every required mode in both train and val.",
    )
    parser.add_argument(
        "--allow_cross_group_pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit same-mode pairs from different source groups inside one partition.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _stable_order_key(seed: int, mode: str, group: str) -> str:
    return hashlib.sha256(f"{seed}:{mode}:{group}".encode("utf-8")).hexdigest()


def _stable_partition(seed: int, mode: str, group: str, validation_fraction: float) -> str:
    digest = _stable_order_key(seed, mode, group)
    unit = int(digest[:16], 16) / float(2**64)
    return "val" if unit < validation_fraction else "train"


def _assign_partitions(
    summaries: list[dict],
    required_modes: tuple[str, ...],
    validation_fraction: float,
    seed: int,
    min_groups_per_partition: int,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be in (0, 1).")

    modes_by_group: dict[str, set[str]] = defaultdict(set)
    for summary in summaries:
        modes_by_group[_comparison_group_key(summary)].add(_command_mode(summary))
    mixed = {group: sorted(modes) for group, modes in modes_by_group.items() if len(modes) != 1}
    if mixed:
        sample = dict(list(sorted(mixed.items()))[:10])
        raise ValueError(
            "Partition-first pairing requires one command mode per stable source group; "
            f"mixed_groups={sample}"
        )

    groups_by_mode: dict[str, list[str]] = defaultdict(list)
    for group, modes in modes_by_group.items():
        groups_by_mode[next(iter(modes))].append(group)

    assignment: dict[str, str] = {}
    stats: dict[str, dict[str, int]] = {}
    for mode in required_modes:
        groups = sorted(groups_by_mode.get(mode, ()))
        required_groups = 2 * min_groups_per_partition
        if len(groups) < required_groups:
            raise ValueError(
                f"Mode {mode} has only {len(groups)} independent source groups; "
                f"need at least {required_groups} to preassign train and validation pools "
                f"with {min_groups_per_partition} groups each."
            )
        partition_by_group = {
            group: _stable_partition(seed, mode, group, validation_fraction) for group in groups
        }
        val_count = sum(value == "val" for value in partition_by_group.values())
        train_count = len(groups) - val_count
        if min(train_count, val_count) < min_groups_per_partition:
            raise ValueError(
                f"Stable V10 partition lacks independent groups for mode={mode}: "
                f"train={train_count}, val={val_count}, required_each={min_groups_per_partition}."
            )
        for group in groups:
            partition = partition_by_group[group]
            previous = assignment.setdefault(group, partition)
            if previous != partition:
                raise RuntimeError(f"Conflicting partition assignment for {group}")
        stats[mode] = {"groups_train": len(groups) - val_count, "groups_val": val_count}

    unknown_modes = sorted(set(groups_by_mode) - set(required_modes))
    for mode in unknown_modes:
        groups = sorted(groups_by_mode[mode])
        for group in groups:
            assignment[group] = _stable_partition(seed, mode, group, validation_fraction)
        val_count = sum(assignment[group] == "val" for group in groups)
        stats[mode] = {"groups_train": len(groups) - val_count, "groups_val": val_count}
    return assignment, stats


def _initial_diversity_scores(summaries: list[dict]) -> np.ndarray:
    features = feature_matrix(summaries, GO2_FEATURE_NAMES).astype(np.float64, copy=False)
    finite = np.where(np.isfinite(features), features, np.nan)
    median = np.nanmedian(finite, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(finite), finite, median)
    scale = np.nanmedian(np.abs(filled - median), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-6), scale, 1.0)
    normalized = np.clip((filled - median) / scale, -20.0, 20.0)
    normalized -= normalized.mean(axis=0, keepdims=True)
    if len(normalized) < 2 or not np.any(np.abs(normalized) > 0.0):
        return np.arange(len(summaries), dtype=np.float64)
    left, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    if not len(singular) or singular[0] <= 1.0e-12:
        return np.arange(len(summaries), dtype=np.float64)
    return left[:, 0] * singular[0]


def _candidate_pairs(
    indices: list[int],
    summaries: list[dict],
    ranking_scores: np.ndarray,
    scorer_active: bool,
    allow_cross_group: bool,
) -> list[tuple[float, int, int]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_group[_comparison_group_key(summaries[index])].append(index)

    pairs: dict[tuple[int, int], tuple[float, int, int]] = {}

    def add(left: int, right: int) -> None:
        if left == right:
            return
        a, b = sorted((int(left), int(right)))
        priority = abs(float(ranking_scores[a] - ranking_scores[b]))
        # Refresh active learning keeps uncertain scorer pairs first.  Initial
        # pretraining has no scorer, so diverse pairs are preferred to avoid
        # ambiguous near-identical stand labels.
        sort_key = priority if scorer_active else -priority
        pairs[(a, b)] = (sort_key, a, b)

    for group_indices in by_group.values():
        order = sorted(group_indices, key=lambda index: float(ranking_scores[index]))
        for position, left in enumerate(order):
            for right in order[position + 1 :]:
                add(left, right)

    if allow_cross_group:
        order = sorted(indices, key=lambda index: float(ranking_scores[index]))
        # Cyclic offsets provide O(N * K) diverse candidates without building
        # an unbounded all-pairs matrix for large 25K datasets.
        max_offsets = min(max(len(order) - 1, 0), 128)
        for offset in range(1, max_offsets + 1):
            for position, left in enumerate(order):
                right = order[(position + offset) % len(order)]
                if _comparison_group_key(summaries[left]) == _comparison_group_key(summaries[right]):
                    continue
                add(left, right)
    return sorted(pairs.values())


def _select_pairs(
    pools: dict[tuple[str, str], list[tuple[float, int, int]]],
    required_modes: tuple[str, ...],
    budget: int,
    minimum: int,
) -> list[tuple[float, int, int, str]]:
    required_total = len(PARTITIONS) * len(required_modes) * minimum
    if budget < required_total:
        raise ValueError(
            f"Pair budget {budget} is below partitioned minimum {required_total} "
            f"({len(required_modes)} modes x 2 partitions x {minimum})."
        )
    selected: list[tuple[float, int, int, str]] = []
    offsets: dict[tuple[str, str], int] = defaultdict(int)
    for partition in PARTITIONS:
        for mode in required_modes:
            key = (partition, mode)
            pool = pools.get(key, [])
            if len(pool) < minimum:
                raise ValueError(
                    f"Insufficient independent pair capacity for partition={partition}, mode={mode}: "
                    f"available={len(pool)}, required={minimum}."
                )
            for item in pool[:minimum]:
                selected.append((*item, partition))
            offsets[key] = minimum

    ordered_keys = [
        (partition, mode)
        for partition in PARTITIONS
        for mode in required_modes
        if pools.get((partition, mode))
    ]
    while len(selected) < budget:
        progressed = False
        for key in ordered_keys:
            offset = offsets[key]
            pool = pools[key]
            if offset < len(pool) and len(selected) < budget:
                selected.append((*pool[offset], key[0]))
                offsets[key] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != budget:
        raise ValueError(f"Could only construct {len(selected)}/{budget} partition-safe feedback pairs.")
    return selected


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be positive.")
    summaries = _read_jsonl(args.summaries)
    references = load_reference_map(args.reference_summaries) if args.reference_summaries else None
    summaries = attach_references_to_summaries(
        summaries,
        references,
        strict=bool(args.reference_summaries),
    )
    if len(summaries) < 2:
        raise ValueError("Need at least two trajectory summaries.")
    required_modes = tuple(args.required_pair_modes) or tuple(
        sorted({_command_mode(summary) for summary in summaries})
    )
    min_groups_per_partition = 2
    if args.allow_cross_group_pairs and args.min_pairs_per_mode_per_partition > 0:
        min_groups_per_partition = max(
            2,
            int(math.ceil((1.0 + math.sqrt(1.0 + 8.0 * args.min_pairs_per_mode_per_partition)) / 2.0)),
        )
    assignment, group_stats = _assign_partitions(
        summaries,
        required_modes,
        args.validation_fraction,
        args.seed,
        min_groups_per_partition,
    )

    scorer_active = bool(args.scorer_checkpoint)
    if scorer_active:
        model, feature_stats, _ = load_scorer_checkpoint(args.scorer_checkpoint)
        ranking_scores = score_summaries(model, summaries, feature_stats).astype(np.float64)
    else:
        ranking_scores = _initial_diversity_scores(summaries)

    indices_by_bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, summary in enumerate(summaries):
        partition = assignment[_comparison_group_key(summary)]
        indices_by_bucket[(partition, _command_mode(summary))].append(index)
    pools = {
        key: _candidate_pairs(
            indices,
            summaries,
            ranking_scores,
            scorer_active,
            args.allow_cross_group_pairs,
        )
        for key, indices in indices_by_bucket.items()
    }
    selected = _select_pairs(
        pools,
        required_modes,
        args.budget,
        args.min_pairs_per_mode_per_partition,
    )

    rng = np.random.default_rng(args.seed)
    rng.shuffle(selected)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".new")
    selected_stats: dict[str, dict[str, int]] = {
        partition: {mode: 0 for mode in required_modes} for partition in PARTITIONS
    }
    with temporary.open("w", encoding="utf-8") as handle:
        for pair_index, (priority, left, right, partition) in enumerate(selected):
            if bool(rng.integers(0, 2)):
                left, right = right, left
            left_group = _comparison_group_key(summaries[left])
            right_group = _comparison_group_key(summaries[right])
            if assignment[left_group] != partition or assignment[right_group] != partition:
                raise RuntimeError("Pair crossed its preassigned split partition.")
            mode = _command_mode(summaries[left])
            if mode != _command_mode(summaries[right]):
                raise RuntimeError("Partitioned pair crossed command modes.")
            selected_stats[partition][mode] = selected_stats[partition].get(mode, 0) + 1
            row = {
                "pair_id": f"{args.pair_prefix}_{pair_index:06d}",
                "pair_score_gap": abs(float(ranking_scores[left] - ranking_scores[right])),
                "pair_priority": float(priority),
                "same_start_state": _start_key(summaries[left]) == _start_key(summaries[right]),
                "split_partition": partition,
                "split_protocol": "stable_group_partition_v1",
                "trajectory_i": summaries[left],
                "trajectory_j": summaries[right],
                "condition": args.condition,
            }
            row["prompt"] = _prompt(row, args.condition)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)

    manifest = {
        "protocol": "stable_group_partition_v1",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "budget": args.budget,
        "required_modes": list(required_modes),
        "min_pairs_per_mode_per_partition": args.min_pairs_per_mode_per_partition,
        "allow_cross_group_pairs": args.allow_cross_group_pairs,
        "scorer_active": scorer_active,
        "group_stats": group_stats,
        "selected_pair_stats": selected_stats,
        "output": str(output),
    }
    manifest_path = Path(args.partition_manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".new")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
