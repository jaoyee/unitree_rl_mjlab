"""MILP selector for exact command quotas using non-overlapping >=40-step intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

from select_go2_matched_blocks_v2 import filter_dataset, stack, time_env


MODE_NAMES = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")
MODE_IDS = {
    (False, False, False): 0, (True, False, False): 1,
    (False, True, False): 2, (False, False, True): 3,
    (True, True, False): 4, (True, False, True): 5,
    (False, True, True): 6, (True, True, True): 7,
}
TARGETS = np.asarray((2000, 6250, 2500, 2000, 3500, 4250, 1250, 3250), dtype=float)
AXES = ("x", "y", "yaw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument("--expert-collector-id", type=int, default=1)
    parser.add_argument("--zero-epsilon", type=float, default=1e-3)
    parser.add_argument("--x-edges", type=float, nargs=4, default=(0.05, 0.20, 0.35, 0.50))
    parser.add_argument("--y-edges", type=float, nargs=4, default=(0.03, 0.087, 0.143, 0.20))
    parser.add_argument("--yaw-edges", type=float, nargs=4, default=(0.05, 0.167, 0.283, 0.40))
    parser.add_argument("--interval-lengths", type=int, nargs="+", default=(40, 50, 60, 80, 100, 120, 160, 200, 240))
    parser.add_argument("--interval-stride", type=int, default=10)
    parser.add_argument("--marginal-target-json", default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument(
        "--mode-tolerance-fraction", type=float, default=0.0,
        help="Per-command-mode quota tolerance; total selected transitions remains exact.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_and_marginals(commands: torch.Tensor, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    values = commands.detach().cpu().numpy()
    shape = values.shape[:2]
    mode_ids = np.full(shape, -1, dtype=np.int8)
    marginals = np.zeros((*shape, 18), dtype=np.int8)
    edges = (args.x_edges, args.y_edges, args.yaw_edges)
    active = np.abs(values) > args.zero_epsilon
    valid = np.ones(shape, dtype=bool)
    buckets = np.full((*shape, 3), -1, dtype=np.int8)
    for axis in range(3):
        magnitude = np.abs(values[..., axis])
        axis_valid = (~active[..., axis]) | ((magnitude >= edges[axis][0]) & (magnitude <= edges[axis][-1] + 1e-6))
        valid &= axis_valid
        bucket = np.digitize(magnitude, edges[axis][1:-1], right=False)
        signed_bucket = bucket + (values[..., axis] > 0).astype(np.int8) * 3
        buckets[..., axis] = np.where(active[..., axis], signed_bucket, -1)
    for bits, mode_id in MODE_IDS.items():
        mask = valid & np.all(active == np.asarray(bits), axis=-1)
        mode_ids[mask] = mode_id
    for axis in range(3):
        for bucket in range(6):
            marginals[..., axis * 6 + bucket] = buckets[..., axis] == bucket
    return mode_ids, marginals


def default_marginal_target() -> np.ndarray:
    axis_totals = np.asarray((17250, 10500, 10750))
    target = []
    for total in axis_totals:
        base, remainder = divmod(int(total), 6)
        target.extend([base + int(index < remainder) for index in range(6)])
    return np.asarray(target, dtype=float)


def candidate_intervals(
    data: dict[str, Any], args: argparse.Namespace
) -> tuple[list[list[tuple[int, int]]], np.ndarray, np.ndarray, tuple[int, int]]:
    commands = time_env(data["commands"], 3)
    episodes = time_env(data["episode_ids"])
    timesteps = time_env(data["timesteps"])
    terminations = time_env(data["terminations"])
    mode_ids, marginal = labels_and_marginals(commands, args)
    allowed = mode_ids >= 0
    allowed &= terminations.detach().cpu().numpy() <= 0.5
    if args.expert_only:
        allowed &= time_env(data["collector_types"]).detach().cpu().numpy() == args.expert_collector_id
    intervals: list[list[tuple[int, int]]] = []
    mode_rows = []
    marginal_rows = []
    lengths = sorted(set(args.interval_lengths))
    for env in range(commands.shape[1]):
        start = 0
        for t in range(1, commands.shape[0] + 1):
            contiguous = (
                t < commands.shape[0] and allowed[t, env] and allowed[t - 1, env]
                and int(episodes[t, env]) == int(episodes[t - 1, env])
                and int(timesteps[t, env]) == int(timesteps[t - 1, env]) + 1
            )
            if contiguous:
                continue
            end = t if allowed[t - 1, env] else t - 1
            if end > start:
                modes = np.eye(8, dtype=np.int16)[mode_ids[start:end, env]]
                mode_prefix = np.vstack((np.zeros((1, 8), dtype=np.int32), modes.cumsum(0)))
                marginal_prefix = np.vstack((np.zeros((1, 18), dtype=np.int32), marginal[start:end, env].cumsum(0)))
                for length in lengths:
                    if end - start < length:
                        continue
                    starts = list(range(start, end - length + 1, args.interval_stride))
                    if starts[-1] != end - length:
                        starts.append(end - length)
                    for window_start in starts:
                        local = window_start - start
                        intervals.append([(source_t, env) for source_t in range(window_start, window_start + length)])
                        mode_rows.append(mode_prefix[local + length] - mode_prefix[local])
                        marginal_rows.append(marginal_prefix[local + length] - marginal_prefix[local])
            start = t
    return intervals, np.asarray(mode_rows), np.asarray(marginal_rows), tuple(commands.shape[:2])


def solve(
    intervals: list[list[tuple[int, int]]], mode_counts: np.ndarray,
    marginal_counts: np.ndarray, shape: tuple[int, int], target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    candidate_count = len(intervals)
    aux_count = 36
    variable_count = candidate_count + aux_count
    exact_mode = np.zeros((8, variable_count), dtype=float)
    exact_mode[:, :candidate_count] = mode_counts.T
    marginal_eq = np.zeros((18, variable_count), dtype=float)
    marginal_eq[:, :candidate_count] = marginal_counts.T
    marginal_eq[:, candidate_count:candidate_count + 18] = -np.eye(18)
    marginal_eq[:, candidate_count + 18:] = np.eye(18)
    rows, cols = [], []
    for candidate, interval in enumerate(intervals):
        for t, env in interval:
            rows.append(t * shape[1] + env)
            cols.append(candidate)
    overlap = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(shape[0] * shape[1], variable_count),
    ).tocsr()
    total = np.zeros((1, variable_count), dtype=float)
    total[0, :candidate_count] = mode_counts.sum(axis=1)
    tolerance = float(args.mode_tolerance_fraction)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("--mode-tolerance-fraction must be in [0, 1)")
    mode_lower = np.ceil(TARGETS * (1.0 - tolerance))
    mode_upper = np.floor(TARGETS * (1.0 + tolerance))
    matrix = vstack(
        (csr_matrix(exact_mode), csr_matrix(total), csr_matrix(marginal_eq), overlap),
        format="csr",
    )
    lower = np.concatenate(
        (mode_lower, [TARGETS.sum()], target, np.full(overlap.shape[0], -np.inf))
    )
    upper = np.concatenate(
        (mode_upper, [TARGETS.sum()], target, np.ones(overlap.shape[0]))
    )
    objective = np.zeros(variable_count)
    objective[:candidate_count] = 1e-4
    weights = 1.0 / np.maximum(target, 1.0)
    objective[candidate_count:candidate_count + 18] = weights
    objective[candidate_count + 18:] = weights
    result = milp(
        c=objective,
        integrality=np.concatenate((np.ones(candidate_count), np.zeros(aux_count))),
        bounds=Bounds(
            np.zeros(variable_count),
            np.concatenate((np.ones(candidate_count), np.full(aux_count, np.inf))),
        ),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": args.time_limit, "mip_rel_gap": args.mip_rel_gap},
    )
    if result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
    selected_ids = np.flatnonzero(result.x[:candidate_count] > 0.5).tolist()
    selected = [intervals[index] for index in selected_ids]
    actual_mode = mode_counts[selected_ids].sum(0)
    if actual_mode.sum() != int(TARGETS.sum()):
        raise RuntimeError(f"MILP returned wrong total count: {actual_mode.sum()}")
    if np.any(actual_mode < mode_lower) or np.any(actual_mode > mode_upper):
        raise RuntimeError(f"MILP returned out-of-tolerance mode counts: {actual_mode.tolist()}")
    return selected, {
        "solver_status": int(result.status), "solver_message": result.message,
        "solver_objective": float(result.fun), "candidate_intervals": candidate_count,
        "selected_interval_ids": selected_ids,
        "actual_mode_counts": actual_mode.tolist(),
        "target_mode_counts": TARGETS.astype(int).tolist(),
        "mode_tolerance_fraction": tolerance,
        "mode_relative_errors": ((actual_mode - TARGETS) / TARGETS).tolist(),
        "actual_marginal_counts": marginal_counts[selected_ids].sum(0).tolist(),
        "target_marginal_counts": target.tolist(),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    intervals, mode_counts, marginal_counts, shape = candidate_intervals(data, args)
    target = default_marginal_target()
    if args.marginal_target_json:
        payload = json.loads(Path(args.marginal_target_json).read_text())
        target = np.asarray(payload["marginal_target_counts"], dtype=float)
    selected, solver = solve(intervals, mode_counts, marginal_counts, shape, target, args)
    output = filter_dataset(data, selected, shape)
    metadata = dict(output.get("metadata") or {})
    metadata.update({
        "condition_id": args.condition_id,
        "selection_kind": "tolerant_mode_nonoverlap_long_interval_milp_v4",
        "selection_source": str(input_path.resolve()),
        "selection_source_sha256": sha256(input_path),
        "selection_minimum_interval": min(args.interval_lengths),
        "selection_interval_lengths": list(args.interval_lengths),
        "selection_zero_epsilon": float(args.zero_epsilon),
        "selection_bin_edges": {"x": args.x_edges, "y": args.y_edges, "yaw": args.yaw_edges},
        "selection_mode_counts": dict(zip(MODE_NAMES, map(int, solver["actual_mode_counts"]), strict=True)),
        "selection_target_mode_counts": dict(zip(MODE_NAMES, map(int, TARGETS), strict=True)),
        "selection_mode_tolerance_fraction": float(args.mode_tolerance_fraction),
        "selection_marginal_counts": solver["actual_marginal_counts"],
        "selection_expert_only": bool(args.expert_only),
        "num_time_steps": 25000, "num_transitions": 25000,
        "num_episodes": len(selected),
    })
    output["metadata"] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report = {
        "schema": "go2_matched_long_interval_selection_v4",
        "condition_id": args.condition_id, "input": str(input_path.resolve()),
        "output": str(output_path.resolve()), "source_shape": list(shape),
        "selected_transitions": sum(map(len, selected)),
        "selected_intervals": len(selected),
        "interval_length_min": min(map(len, selected)),
        "interval_length_max": max(map(len, selected)),
        "valid_40_step_starts": sum(len(block) - 39 for block in selected),
        "valid_100_step_starts": sum(max(0, len(block) - 99) for block in selected),
        **solver,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "condition_id", "selected_transitions", "selected_intervals",
        "valid_40_step_starts", "solver_status", "solver_message",
    )}, indent=2))


if __name__ == "__main__":
    main()
