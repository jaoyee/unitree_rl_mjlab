"""Exact-quota selector with a minimum 40-step selected-run constraint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from select_go2_matched_blocks_v2 import filter_dataset, time_env
from select_go2_matched_intervals_v2 import (
    AXES, MODE_NAMES, TARGETS, default_marginal_target, labels_and_marginals, sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--minimum-run", type=int, default=40)
    parser.add_argument("--enforce-minimum-run", action="store_true")
    parser.add_argument("--start-penalty", type=float, default=1e-3)
    parser.add_argument("--zero-epsilon", type=float, default=1e-3)
    parser.add_argument("--x-edges", type=float, nargs=4, default=(0.05, 0.20, 0.35, 0.50))
    parser.add_argument("--y-edges", type=float, nargs=4, default=(0.03, 0.087, 0.143, 0.20))
    parser.add_argument("--yaw-edges", type=float, nargs=4, default=(0.05, 0.167, 0.283, 0.40))
    parser.add_argument("--marginal-target-json", default=None)
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    return parser.parse_args()


def flatten_env_major(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return np.swapaxes(array, 0, 1).reshape(-1, *array.shape[2:])


def build_segments(allowed: np.ndarray, episodes: np.ndarray, timesteps: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = 0
    for index in range(1, len(allowed) + 1):
        contiguous = (
            index < len(allowed) and allowed[index] and allowed[index - 1]
            and episodes[index] == episodes[index - 1]
            and timesteps[index] == timesteps[index - 1] + 1
        )
        if contiguous:
            continue
        end = index if allowed[index - 1] else index - 1
        if end > start:
            segments.append((start, end))
        start = index
    return segments


def solve(data: dict, args: argparse.Namespace) -> tuple[list[list[tuple[int, int]]], dict]:
    commands_te = time_env(data["commands"], 3)
    modes_te, marginals_te = labels_and_marginals(commands_te, args)
    terminations = flatten_env_major(time_env(data["terminations"])).reshape(-1)
    episodes = flatten_env_major(time_env(data["episode_ids"])).reshape(-1)
    timesteps = flatten_env_major(time_env(data["timesteps"])).reshape(-1)
    modes = np.swapaxes(modes_te, 0, 1).reshape(-1)
    marginals = np.swapaxes(marginals_te, 0, 1).reshape(-1, 18)
    allowed = (modes >= 0) & (terminations <= 0.5)
    segments = build_segments(allowed, episodes, timesteps)
    n = len(modes)
    aux_offset = 2 * n
    variables = aux_offset + 36
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for mode_id, target in enumerate(TARGETS):
        indices = np.flatnonzero(modes == mode_id)
        row_indices.extend([row] * len(indices)); col_indices.extend(indices.tolist()); values.extend([1.0] * len(indices))
        lower.append(float(target)); upper.append(float(target)); row += 1
    target_marginal = default_marginal_target()
    if args.marginal_target_json:
        target_marginal = np.asarray(json.loads(Path(args.marginal_target_json).read_text())["marginal_target_counts"], dtype=float)
    for marginal_id in range(18):
        indices = np.flatnonzero(marginals[:, marginal_id])
        row_indices.extend([row] * len(indices)); col_indices.extend(indices.tolist()); values.extend([1.0] * len(indices))
        row_indices.extend((row, row)); col_indices.extend((aux_offset + marginal_id, aux_offset + 18 + marginal_id)); values.extend((-1.0, 1.0))
        lower.append(float(target_marginal[marginal_id])); upper.append(float(target_marginal[marginal_id])); row += 1
    horizon = int(args.minimum_run)
    for start, end in segments:
        for index in range(start, end):
            # x[index] - x[index-1] - start[index] <= 0; previous x is zero at a segment boundary.
            row_indices.extend((row, row)); col_indices.extend((index, n + index)); values.extend((1.0, -1.0))
            if index > start:
                row_indices.append(row); col_indices.append(index - 1); values.append(-1.0)
            lower.append(-np.inf); upper.append(0.0); row += 1
            if args.enforce_minimum_run:
                # A selected run start requires the next `horizon` samples in this source segment.
                stop = min(end, index + horizon)
                row_indices.append(row); col_indices.append(n + index); values.append(float(horizon))
                for future in range(index, stop):
                    row_indices.append(row); col_indices.append(future); values.append(-1.0)
                lower.append(-np.inf); upper.append(0.0); row += 1
    matrix = coo_matrix((values, (row_indices, col_indices)), shape=(row, variables)).tocsr()
    upper_bounds = np.concatenate((allowed.astype(float), allowed.astype(float), np.full(36, np.inf)))
    objective = np.zeros(variables)
    objective[n:2 * n] = float(args.start_penalty)
    weights = 1.0 / np.maximum(target_marginal, 1.0)
    objective[aux_offset:aux_offset + 18] = weights
    objective[aux_offset + 18:] = weights
    result = milp(
        c=objective,
        integrality=np.concatenate((np.ones(2 * n), np.zeros(36))),
        bounds=Bounds(np.zeros(variables), upper_bounds),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": args.time_limit, "mip_rel_gap": args.mip_rel_gap},
    )
    if result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
    selected = result.x[:n] > 0.5
    blocks_flat = []
    for start, end in segments:
        index = start
        while index < end:
            if not selected[index]:
                index += 1
                continue
            run_end = index + 1
            while run_end < end and selected[run_end]:
                run_end += 1
            blocks_flat.append(list(range(index, run_end)))
            index = run_end
    if args.enforce_minimum_run and min(map(len, blocks_flat), default=0) < horizon:
        raise RuntimeError("solver output violates minimum selected-run length")
    time_steps, num_envs = commands_te.shape[:2]
    blocks = [
        [(flat % time_steps, flat // time_steps) for flat in block]
        for block in blocks_flat
    ]
    selected_ids = np.flatnonzero(selected)
    return blocks, {
        "solver_status": int(result.status), "solver_message": result.message,
        "solver_objective": float(result.fun), "source_segments": len(segments),
        "actual_mode_counts": np.bincount(modes[selected_ids], minlength=8).tolist(),
        "actual_marginal_counts": marginals[selected_ids].sum(0).tolist(),
        "target_marginal_counts": target_marginal.tolist(),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    blocks, solver = solve(data, args)
    shape = tuple(time_env(data["commands"], 3).shape[:2])
    output = filter_dataset(data, blocks, shape)
    metadata = dict(output.get("metadata") or {})
    metadata.update({
        "condition_id": args.condition_id,
        "selection_kind": (
            "exact_mode_minimum_run_sequence_milp_v2"
            if args.enforce_minimum_run
            else "exact_mode_continuity_optimized_sequence_milp_v2"
        ),
        "selection_source": str(input_path.resolve()),
        "selection_source_sha256": sha256(input_path),
        "selection_minimum_run": int(args.minimum_run),
        "selection_minimum_run_enforced": bool(args.enforce_minimum_run),
        "selection_continuity_objective": "minimize_selected_run_starts",
        "selection_mode_counts": dict(zip(MODE_NAMES, map(int, TARGETS), strict=True)),
        "selection_marginal_counts": solver["actual_marginal_counts"],
        "selection_zero_epsilon": float(args.zero_epsilon),
        "selection_bin_edges": {"x": args.x_edges, "y": args.y_edges, "yaw": args.yaw_edges},
        "num_time_steps": 25000, "num_transitions": 25000, "num_episodes": len(blocks),
    })
    output["metadata"] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report = {
        "schema": "go2_matched_sequence_selection_v2", "condition_id": args.condition_id,
        "input": str(input_path.resolve()), "output": str(output_path.resolve()),
        "selected_transitions": sum(map(len, blocks)), "selected_runs": len(blocks),
        "run_length_min": min(map(len, blocks)), "run_length_max": max(map(len, blocks)),
        "valid_40_step_starts": sum(len(block) - 39 for block in blocks), **solver,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "condition_id", "selected_transitions", "selected_runs", "run_length_min",
        "valid_40_step_starts", "solver_status", "solver_message",
    )}, indent=2))


if __name__ == "__main__":
    main()
