"""Select 25K from disjoint 40-step blocks, then merge adjacent selected blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_go2_matched_blocks_v2 import filter_dataset, time_env
from select_go2_matched_intervals_v2 import MODE_NAMES, default_marginal_target, labels_and_marginals


TARGETS = np.asarray((2000, 6250, 2500, 2000, 3500, 4250, 1250, 3250), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--block-length", type=int, default=40)
    parser.add_argument("--quota-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=300.0)
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


def partition_blocks(data: dict[str, Any], args: argparse.Namespace):
    commands = time_env(data["commands"], 3)
    episodes = time_env(data["episode_ids"])
    timesteps = time_env(data["timesteps"])
    terminations = time_env(data["terminations"])
    mode_ids, marginals = labels_and_marginals(commands, args)
    allowed = (mode_ids >= 0) & (terminations.detach().cpu().numpy() <= 0.5)
    available_counts = np.bincount(mode_ids[allowed], minlength=8).astype(np.float64)
    scarcity_weights = TARGETS / np.maximum(available_counts, 1.0)
    blocks: list[list[tuple[int, int]]] = []
    block_runs: list[int] = []
    mode_rows: list[np.ndarray] = []
    marginal_rows: list[np.ndarray] = []
    run_id = 0
    for env in range(commands.shape[1]):
        start = 0
        for t in range(1, commands.shape[0] + 1):
            contiguous = (
                t < commands.shape[0] and allowed[t - 1, env] and allowed[t, env]
                and int(episodes[t, env]) == int(episodes[t - 1, env])
                and int(timesteps[t, env]) == int(timesteps[t - 1, env]) + 1
            )
            if contiguous:
                continue
            if allowed[t - 1, env]:
                stop = t
                run_length = stop - start
                usable = (run_length // args.block_length) * args.block_length
                discard = run_length - usable
                # Choose which <=39 boundary rows to discard. Keep scarce modes
                # (for example g0 pure_y) and discard rows from abundant modes.
                best_drop = 0
                best_discard_cost = float("inf")
                for prefix_drop in range(discard + 1):
                    suffix_drop = discard - prefix_drop
                    discarded = np.concatenate((
                        mode_ids[start:start + prefix_drop, env],
                        mode_ids[stop - suffix_drop:stop, env] if suffix_drop else np.empty(0, dtype=np.int8),
                    ))
                    cost = float(scarcity_weights[discarded].sum()) if discarded.size else 0.0
                    if cost < best_discard_cost:
                        best_discard_cost = cost
                        best_drop = prefix_drop
                cursor = start + best_drop
                usable_stop = cursor + usable
                while usable_stop - cursor >= args.block_length:
                    end = cursor + args.block_length
                    blocks.append([(source_t, env) for source_t in range(cursor, end)])
                    block_runs.append(run_id)
                    mode_rows.append(np.bincount(mode_ids[cursor:end, env], minlength=8))
                    marginal_rows.append(marginals[cursor:end, env].sum(axis=0))
                    cursor = end
                run_id += 1
            start = t
    return (
        blocks, np.asarray(block_runs), np.asarray(mode_rows, dtype=np.int64),
        np.asarray(marginal_rows, dtype=np.int64), tuple(commands.shape[:2]),
    )


def solve_blocks(blocks, block_runs, mode_counts, marginal_counts, args):
    n = len(blocks)
    needed_blocks, remainder = divmod(int(TARGETS.sum()), args.block_length)
    if remainder:
        raise ValueError("Target total must be divisible by block length")
    if n < needed_blocks:
        raise RuntimeError(f"Only {n} eligible blocks; need {needed_blocks}")
    tolerance = args.quota_tolerance_fraction
    lower_modes = np.ceil(TARGETS * (1.0 - tolerance))
    upper_modes = np.floor(TARGETS * (1.0 + tolerance))
    marginal_target = default_marginal_target()
    aux = 36
    variables = n + aux
    mode_matrix = np.zeros((8, variables))
    mode_matrix[:, :n] = mode_counts.T
    count_matrix = np.zeros((1, variables))
    count_matrix[0, :n] = 1.0
    marginal_matrix = np.zeros((18, variables))
    marginal_matrix[:, :n] = marginal_counts.T
    marginal_matrix[:, n:n + 18] = -np.eye(18)
    marginal_matrix[:, n + 18:] = np.eye(18)
    matrix = vstack(
        (csr_matrix(mode_matrix), csr_matrix(count_matrix), csr_matrix(marginal_matrix)),
        format="csr",
    )
    lower = np.concatenate((lower_modes, [needed_blocks], marginal_target))
    upper = np.concatenate((upper_modes, [needed_blocks], marginal_target))
    objective = np.zeros(variables)
    rng = np.random.default_rng(args.seed)
    # A deterministic tiny tie-breaker; marginal balance dominates this term.
    objective[:n] = rng.uniform(0.0, 1e-8, size=n)
    weights = 1.0 / np.maximum(marginal_target, 1.0)
    objective[n:n + 18] = weights
    objective[n + 18:] = weights
    result = milp(
        c=objective,
        integrality=np.concatenate((np.ones(n), np.zeros(aux))),
        bounds=Bounds(np.zeros(variables), np.concatenate((np.ones(n), np.full(aux, np.inf)))),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": args.time_limit, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        capacity = mode_counts.sum(axis=0).astype(int).tolist()
        raise RuntimeError(
            f"Partitioned block solve failed: status={result.status}, message={result.message}, "
            f"mode_capacity={dict(zip(MODE_NAMES, capacity, strict=True))}"
        )
    ids = np.flatnonzero(result.x[:n] > 0.5)
    if len(ids) != needed_blocks:
        raise RuntimeError(f"Wrong selected block count: {len(ids)}")
    return ids, {
        "solver_status": int(result.status),
        "solver_message": result.message,
        "candidate_blocks": n,
        "selected_blocks": len(ids),
        "actual_mode_counts": mode_counts[ids].sum(axis=0).astype(int).tolist(),
        "desired_mode_counts": TARGETS.astype(int).tolist(),
        "mode_relative_errors": ((mode_counts[ids].sum(axis=0) - TARGETS) / TARGETS).tolist(),
        "actual_marginal_counts": marginal_counts[ids].sum(axis=0).astype(int).tolist(),
        "target_marginal_counts": marginal_target.astype(int).tolist(),
    }


def merge_adjacent(blocks, block_runs, selected_ids):
    ordered = sorted(selected_ids, key=lambda index: (block_runs[index], blocks[index][0]))
    merged: list[list[tuple[int, int]]] = []
    last_run = None
    for index in ordered:
        block = blocks[index]
        run = int(block_runs[index])
        if merged and run == last_run and merged[-1][-1][1] == block[0][1] and merged[-1][-1][0] + 1 == block[0][0]:
            merged[-1].extend(block)
        else:
            merged.append(list(block))
        last_run = run
    return merged


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    blocks, block_runs, mode_counts, marginal_counts, shape = partition_blocks(data, args)
    selected_ids, solver = solve_blocks(blocks, block_runs, mode_counts, marginal_counts, args)
    selected = merge_adjacent(blocks, block_runs, selected_ids)
    output = filter_dataset(data, selected, shape)
    lengths = [len(interval) for interval in selected]
    actual_total = sum(lengths)
    if actual_total != int(TARGETS.sum()):
        raise RuntimeError(f"Wrong selected total: {actual_total}")
    metadata = dict(output.get("metadata") or {})
    metadata.update({
        "condition_id": args.condition_id,
        "selection_kind": "disjoint_partitioned_40step_blocks_milp_v4",
        "selection_source": str(input_path.resolve()),
        "selection_source_sha256": sha256(input_path),
        "selection_block_length": int(args.block_length),
        "selection_quota_tolerance_fraction": float(args.quota_tolerance_fraction),
        "selection_desired_mode_counts": dict(zip(MODE_NAMES, TARGETS.astype(int).tolist(), strict=True)),
        "selection_actual_mode_counts": dict(zip(MODE_NAMES, solver["actual_mode_counts"], strict=True)),
        "selection_zero_epsilon": float(args.zero_epsilon),
        "selection_bin_edges": {"x": args.x_edges, "y": args.y_edges, "yaw": args.yaw_edges},
        "num_transitions": actual_total,
        "num_episodes": len(selected),
    })
    output["metadata"] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report = {
        "schema": "go2_partitioned_block_selection_v4",
        "condition_id": args.condition_id,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "source_shape": list(shape),
        "selected_transitions": actual_total,
        "merged_intervals": len(selected),
        "interval_length_min": min(lengths),
        "interval_length_median": float(np.median(lengths)),
        "interval_length_max": max(lengths),
        "valid_40_step_starts": sum(length - 39 for length in lengths),
        "valid_100_step_starts": sum(max(0, length - 99) for length in lengths),
        "valid_200_step_starts": sum(max(0, length - 199) for length in lengths),
        **solver,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
