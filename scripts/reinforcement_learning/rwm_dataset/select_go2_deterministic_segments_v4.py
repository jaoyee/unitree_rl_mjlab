"""Deterministically select an exact 25K command-balanced dataset from contiguous runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_go2_matched_blocks_v2 import filter_dataset, time_env
from select_go2_matched_intervals_v2 import MODE_IDS, MODE_NAMES, labels_and_marginals


DEFAULT_TARGETS = (2000, 6250, 2500, 2000, 3500, 4250, 1250, 3250)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--target-counts", type=int, nargs=8, default=DEFAULT_TARGETS)
    parser.add_argument("--quota-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-interval", type=int, default=40)
    parser.add_argument("--maximum-interval", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument("--expert-collector-id", type=int, default=1)
    parser.add_argument("--zero-epsilon", type=float, default=1e-3)
    parser.add_argument("--x-edges", type=float, nargs=4, default=(0.001, 0.20, 0.35, 0.50))
    parser.add_argument("--y-edges", type=float, nargs=4, default=(0.001, 0.087, 0.143, 0.20))
    parser.add_argument("--yaw-edges", type=float, nargs=4, default=(0.001, 0.167, 0.283, 0.40))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_run(start: int, stop: int, env: int, minimum: int, maximum: int) -> list[list[tuple[int, int]]]:
    """Split a run into non-overlapping chunks, never leaving a sub-minimum tail."""
    chunks: list[list[tuple[int, int]]] = []
    cursor = start
    while stop - cursor > maximum:
        length = maximum
        tail = stop - (cursor + length)
        if 0 < tail < minimum:
            length -= minimum - tail
        chunks.append([(t, env) for t in range(cursor, cursor + length)])
        cursor += length
    if stop - cursor >= minimum:
        chunks.append([(t, env) for t in range(cursor, stop)])
    return chunks


def candidate_chunks(
    data: dict[str, Any], args: argparse.Namespace
) -> tuple[list[list[list[tuple[int, int]]]], tuple[int, int], np.ndarray]:
    commands = time_env(data["commands"], 3)
    episodes = time_env(data["episode_ids"])
    timesteps = time_env(data["timesteps"])
    terminations = time_env(data["terminations"])
    mode_ids, _ = labels_and_marginals(commands, args)
    allowed = mode_ids >= 0
    allowed &= terminations.detach().cpu().numpy() <= 0.5
    if args.expert_only:
        if "collector_types" not in data:
            raise KeyError("--expert-only requested but collector_types is absent")
        allowed &= time_env(data["collector_types"]).detach().cpu().numpy() == args.expert_collector_id

    by_mode: list[list[list[tuple[int, int]]]] = [[] for _ in MODE_NAMES]
    for env in range(commands.shape[1]):
        start = 0
        for t in range(1, commands.shape[0] + 1):
            same_run = (
                t < commands.shape[0]
                and allowed[t - 1, env]
                and allowed[t, env]
                and mode_ids[t, env] == mode_ids[t - 1, env]
                and int(episodes[t, env]) == int(episodes[t - 1, env])
                and int(timesteps[t, env]) == int(timesteps[t - 1, env]) + 1
            )
            if same_run:
                continue
            if allowed[t - 1, env]:
                stop = t
                mode = int(mode_ids[t - 1, env])
                if stop - start >= args.minimum_interval:
                    by_mode[mode].extend(
                        split_run(start, stop, env, args.minimum_interval, args.maximum_interval)
                    )
            start = t
    return by_mode, tuple(commands.shape[:2]), mode_ids


def select_exact(
    candidates: list[list[list[tuple[int, int]]]], targets: np.ndarray,
    minimum: int, seed: int,
) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[list[tuple[int, int]]] = []
    report: dict[str, Any] = {"per_mode": {}}
    for mode, name in enumerate(MODE_NAMES):
        pool = list(candidates[mode])
        rng.shuffle(pool)
        pool.sort(key=len, reverse=True)
        capacity = sum(map(len, pool))
        target = int(targets[mode])
        if capacity < target:
            raise RuntimeError(
                f"Insufficient >= {minimum}-step capacity for {name}: capacity={capacity}, target={target}"
            )
        # Select the smallest number of longest chunks whose combined capacity
        # covers the quota.  Then give every selected chunk the minimum legal
        # length and distribute the remaining rows within their capacities.
        # The former one-pass greedy method could strand a remainder smaller
        # than ``minimum`` even when an exact solution plainly existed.
        selected_pool: list[list[tuple[int, int]]] = []
        selected_capacity = 0
        for chunk in pool:
            selected_pool.append(chunk)
            selected_capacity += len(chunk)
            if selected_capacity >= target:
                break
        if selected_capacity < target or len(selected_pool) * minimum > target:
            raise RuntimeError(
                f"Could not compose exact quota for {name}: target={target}, "
                f"selected_chunks={len(selected_pool)}, selected_capacity={selected_capacity}, "
                f"minimum_required={len(selected_pool) * minimum}, total_capacity={capacity}"
            )
        take_lengths = [minimum] * len(selected_pool)
        remaining = target - sum(take_lengths)
        for index, chunk in enumerate(selected_pool):
            extra = min(remaining, len(chunk) - minimum)
            take_lengths[index] += extra
            remaining -= extra
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(
                f"Internal exact-allocation failure for {name}: target={target}, remaining={remaining}"
            )
        chosen = [chunk[:take] for chunk, take in zip(selected_pool, take_lengths, strict=True)]
        selected.extend(chosen)
        lengths = [len(chunk) for chunk in chosen]
        report["per_mode"][name] = {
            "target": target,
            "selected": sum(lengths),
            "available_capacity": capacity,
            "selected_intervals": len(lengths),
            "interval_min": min(lengths),
            "interval_median": float(np.median(lengths)),
            "interval_max": max(lengths),
            "valid_100_step_starts": sum(max(0, length - 99) for length in lengths),
        }
    rng.shuffle(selected)
    return selected, report


def capacity_adjusted_targets(
    candidates: list[list[list[tuple[int, int]]]], desired: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("quota tolerance must be in [0, 1)")
    desired = np.asarray(desired, dtype=np.int64)
    capacities = np.asarray([sum(map(len, pool)) for pool in candidates], dtype=np.int64)
    lower = np.ceil(desired * (1.0 - tolerance)).astype(np.int64)
    upper = np.floor(desired * (1.0 + tolerance)).astype(np.int64)
    insufficient = {
        MODE_NAMES[index]: {"capacity": int(capacities[index]), "minimum": int(lower[index])}
        for index in range(8) if capacities[index] < lower[index]
    }
    if insufficient:
        raise RuntimeError(f"Insufficient per-mode capacity even with tolerance: {insufficient}")
    adjusted = np.minimum(desired, capacities)
    deficit = int(desired.sum() - adjusted.sum())
    headroom = np.minimum(capacities, upper) - adjusted
    for index in np.argsort(-headroom):
        add = min(deficit, int(headroom[index]))
        adjusted[index] += add
        deficit -= add
        if deficit == 0:
            break
    if deficit:
        raise RuntimeError(
            f"Could not redistribute {deficit} transitions within tolerance; "
            f"capacities={capacities.tolist()}, desired={desired.tolist()}"
        )
    return adjusted, capacities


def main() -> None:
    args = parse_args()
    if args.minimum_interval < 2 or args.maximum_interval < args.minimum_interval:
        raise ValueError("Require 2 <= minimum_interval <= maximum_interval")
    desired_targets = np.asarray(args.target_counts, dtype=np.int64)
    input_path = Path(args.input).expanduser()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    candidates, shape, _ = candidate_chunks(data, args)
    targets, capacities = capacity_adjusted_targets(
        candidates, desired_targets, args.quota_tolerance_fraction
    )
    selected, details = select_exact(candidates, targets, args.minimum_interval, args.seed)
    output = filter_dataset(data, selected, shape)
    lengths = [len(interval) for interval in selected]
    actual_total = sum(lengths)
    if actual_total != int(targets.sum()):
        raise RuntimeError(f"Wrong selected total: {actual_total} != {targets.sum()}")

    metadata = dict(output.get("metadata") or {})
    metadata.update({
        "condition_id": args.condition_id,
        "selection_kind": "deterministic_pure_mode_contiguous_segments_v4",
        "selection_source": str(input_path.resolve()),
        "selection_source_sha256": sha256(input_path),
        "selection_seed": int(args.seed),
        "selection_minimum_interval": int(args.minimum_interval),
        "selection_maximum_interval": int(args.maximum_interval),
        "selection_zero_epsilon": float(args.zero_epsilon),
        "selection_bin_edges": {"x": args.x_edges, "y": args.y_edges, "yaw": args.yaw_edges},
        "selection_mode_counts": dict(zip(MODE_NAMES, targets.astype(int).tolist(), strict=True)),
        "selection_desired_mode_counts": dict(
            zip(MODE_NAMES, desired_targets.astype(int).tolist(), strict=True)
        ),
        "selection_quota_tolerance_fraction": float(args.quota_tolerance_fraction),
        "selection_expert_only": bool(args.expert_only),
        "num_transitions": actual_total,
        "num_episodes": len(selected),
    })
    output["metadata"] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    report = {
        "schema": "go2_deterministic_segment_selection_v4",
        "condition_id": args.condition_id,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "source_shape": list(shape),
        "selected_transitions": actual_total,
        "desired_mode_counts": dict(zip(MODE_NAMES, desired_targets.astype(int).tolist(), strict=True)),
        "actual_mode_counts": dict(zip(MODE_NAMES, targets.astype(int).tolist(), strict=True)),
        "available_mode_capacity": dict(zip(MODE_NAMES, capacities.astype(int).tolist(), strict=True)),
        "selected_intervals": len(selected),
        "interval_length_min": min(lengths),
        "interval_length_median": float(np.median(lengths)),
        "interval_length_max": max(lengths),
        "valid_40_step_starts": sum(length - 39 for length in lengths),
        "valid_100_step_starts": sum(max(0, length - 99) for length in lengths),
        "valid_200_step_starts": sum(max(0, length - 199) for length in lengths),
        **details,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "condition_id": args.condition_id,
        "selected_transitions": actual_total,
        "selected_intervals": len(selected),
        "interval_length_min": min(lengths),
        "interval_length_median": float(np.median(lengths)),
        "valid_100_step_starts": report["valid_100_step_starts"],
    }, indent=2))


if __name__ == "__main__":
    main()
