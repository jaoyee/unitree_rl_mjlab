"""Distribution-constrained global trajectory selection for Go2 TRACE V10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_trace.v10_protocol import (
    AXES,
    COMMAND_MODES,
    MODE_TO_ID,
    classify_command,
    load_protocol,
    marginal_vector,
    signed_magnitude_buckets,
)


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Largest-remainder weights must be a non-empty non-negative vector.")
    raw = total * weights / weights.sum()
    result = np.floor(raw).astype(np.int64)
    remainder = int(total - result.sum())
    order = sorted(range(len(weights)), key=lambda index: (-(raw[index] - result[index]), index))
    for index in order[:remainder]:
        result[index] += 1
    return result


def dataset_marginal_targets(
    dataset_path: str | Path,
    protocol: Mapping[str, Any],
    selected_mode_counts: Mapping[str, int],
) -> np.ndarray:
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    commands = stack_time_key(dataset, "commands").reshape(-1, 3).detach().cpu().numpy()
    source = np.zeros(18, dtype=np.int64)
    for command in commands:
        source += marginal_vector(command, protocol)

    active_totals = {
        "x": sum(int(selected_mode_counts[mode]) for mode in ("pure_x", "xy", "x_yaw", "xy_yaw")),
        "y": sum(int(selected_mode_counts[mode]) for mode in ("pure_y", "xy", "y_yaw", "xy_yaw")),
        "yaw": sum(int(selected_mode_counts[mode]) for mode in ("pure_yaw", "x_yaw", "y_yaw", "xy_yaw")),
    }
    target = np.zeros(18, dtype=np.int64)
    for axis_index, axis in enumerate(AXES):
        axis_source = source[axis_index * 6 : (axis_index + 1) * 6]
        if axis_source.sum() <= 0:
            raise ValueError(f"Dataset contains no active commands for axis {axis}.")
        target[axis_index * 6 : (axis_index + 1) * 6] = _largest_remainder(
            active_totals[axis], axis_source.astype(np.float64)
        )
    return target


def dataset_mode_bucket_probabilities(
    dataset_path: str | Path, protocol: Mapping[str, Any]
) -> list[list[list[float]]]:
    """Return `[mode, axis, signed_bucket]` probabilities from the immutable 25K dataset."""

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    commands = stack_time_key(dataset, "commands").reshape(-1, 3).detach().cpu().numpy()
    counts = np.zeros((len(COMMAND_MODES), 3, 6), dtype=np.float64)
    for command in commands:
        mode_id = MODE_TO_ID[classify_command(command)]
        signed = signed_magnitude_buckets(command, protocol)
        for axis, bucket in enumerate(signed):
            if bucket >= 0:
                counts[mode_id, axis, bucket] += 1.0
    probabilities = np.zeros_like(counts)
    for mode_id in range(len(COMMAND_MODES)):
        for axis in range(3):
            total = counts[mode_id, axis].sum()
            if total > 0.0:
                probabilities[mode_id, axis] = counts[mode_id, axis] / total
    return probabilities.tolist()


def _summary_command(summary: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [summary.get("command_vx_mean"), summary.get("command_vy_mean"), summary.get("command_yaw_mean")],
        dtype=np.float64,
    )


def select_v10_trajectories(
    summaries: Sequence[dict[str, Any]],
    scores: np.ndarray,
    eligible_indices: np.ndarray,
    *,
    protocol: Mapping[str, Any],
    marginal_targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the globally best scorer set after exact command-distribution constraints."""

    eligible = np.asarray(eligible_indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(summaries),):
        raise ValueError(f"Expected one scorer value per summary, got {scores.shape} for {len(summaries)} rows.")
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("Eligible trajectory indices must be unique.")
    candidate_cfg = protocol["candidate"]
    target_by_mode = {mode: int(candidate_cfg["selected_mode_counts"][mode]) for mode in COMMAND_MODES}
    target_count = int(candidate_cfg["selected_trajectory_count"])
    if sum(target_by_mode.values()) != target_count:
        raise ValueError("Selected mode targets do not sum to selected_trajectory_count.")

    modes = np.empty(len(eligible), dtype=np.int8)
    marginals = np.zeros((len(eligible), 18), dtype=np.int8)
    for local_index, global_index in enumerate(eligible):
        command = _summary_command(summaries[int(global_index)])
        mode = str(summaries[int(global_index)].get("command_mode") or classify_command(command))
        if mode not in MODE_TO_ID:
            raise ValueError(f"Unknown command mode {mode!r} at summary {global_index}.")
        modes[local_index] = MODE_TO_ID[mode]
        marginals[local_index] = marginal_vector(command, protocol)

    availability = {mode: int(np.sum(modes == MODE_TO_ID[mode])) for mode in COMMAND_MODES}
    deficits = {
        mode: {"available": availability[mode], "required": target_by_mode[mode]}
        for mode in COMMAND_MODES
        if availability[mode] < target_by_mode[mode]
    }
    if deficits:
        raise ValueError(f"V10 eligible pool cannot satisfy exact command-mode quotas: {deficits}")

    n = len(eligible)
    slack_offset = n
    variable_count = n + 36
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for mode in COMMAND_MODES:
        local_indices = np.flatnonzero(modes == MODE_TO_ID[mode])
        rows.extend([row] * len(local_indices))
        cols.extend(local_indices.tolist())
        values.extend([1.0] * len(local_indices))
        target = float(target_by_mode[mode])
        lower.append(target)
        upper.append(target)
        row += 1
    marginal_targets = np.asarray(marginal_targets, dtype=np.float64).reshape(18)
    for marginal_index in range(18):
        local_indices = np.flatnonzero(marginals[:, marginal_index])
        rows.extend([row] * len(local_indices))
        cols.extend(local_indices.tolist())
        values.extend([1.0] * len(local_indices))
        rows.extend((row, row))
        cols.extend((slack_offset + marginal_index, slack_offset + 18 + marginal_index))
        values.extend((-1.0, 1.0))
        target = float(marginal_targets[marginal_index])
        lower.append(target)
        upper.append(target)
        row += 1
    constraints = coo_matrix((values, (rows, cols)), shape=(row, variable_count)).tocsr()
    integrality = np.concatenate((np.ones(n, dtype=np.int8), np.zeros(36, dtype=np.int8)))
    bounds = Bounds(np.zeros(variable_count), np.concatenate((np.ones(n), np.full(36, np.inf))))
    options = {
        "time_limit": float(candidate_cfg["milp_time_limit_seconds"]),
        "mip_rel_gap": float(candidate_cfg["milp_relative_gap"]),
    }

    slack_weights = 1.0 / np.maximum(marginal_targets, 1.0)
    distribution_objective = np.zeros(variable_count, dtype=np.float64)
    distribution_objective[slack_offset : slack_offset + 18] = slack_weights
    distribution_objective[slack_offset + 18 :] = slack_weights
    first = milp(
        c=distribution_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(constraints, np.asarray(lower), np.asarray(upper)),
        options=options,
    )
    if first.x is None:
        raise RuntimeError(f"V10 distribution MILP failed: status={first.status}, message={first.message}")
    optimum_distribution_error = float(distribution_objective @ first.x)

    order = np.argsort(scores[eligible], kind="stable")
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.linspace(0.0, 1.0, n, endpoint=True)
    score_objective = np.zeros(variable_count, dtype=np.float64)
    score_objective[:n] = -rank
    distribution_row = csr_matrix(distribution_objective.reshape(1, -1))
    second_constraints = vstack((constraints, distribution_row), format="csr")
    second_lower = np.concatenate((np.asarray(lower), [-np.inf]))
    second_upper = np.concatenate((np.asarray(upper), [optimum_distribution_error + 1.0e-6]))
    second = milp(
        c=score_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(second_constraints, second_lower, second_upper),
        options=options,
    )
    if second.x is None:
        raise RuntimeError(f"V10 scorer MILP failed: status={second.status}, message={second.message}")
    selected_local = np.flatnonzero(second.x[:n] > 0.5)
    if len(selected_local) != target_count:
        raise RuntimeError(f"V10 selector returned {len(selected_local)} trajectories, expected {target_count}.")
    selected = eligible[selected_local]
    selected_modes = modes[selected_local]
    selected_marginals = marginals[selected_local].sum(axis=0)
    actual_by_mode = {mode: int(np.sum(selected_modes == MODE_TO_ID[mode])) for mode in COMMAND_MODES}
    if actual_by_mode != target_by_mode:
        raise RuntimeError(f"V10 selector violated exact mode quotas: {actual_by_mode} != {target_by_mode}")
    normalized_l1 = float(
        np.sum(np.abs(selected_marginals - marginal_targets)) / max(float(marginal_targets.sum()), 1.0)
    )
    diagnostics = {
        "selection_protocol": "distribution_constrained_global_top25_v10",
        "eligible_count": int(len(eligible)),
        "selected_count": int(len(selected)),
        "availability_by_mode": availability,
        "target_mode_counts": target_by_mode,
        "selected_mode_counts": actual_by_mode,
        "target_signed_marginal_counts": marginal_targets.astype(int).tolist(),
        "selected_signed_marginal_counts": selected_marginals.astype(int).tolist(),
        "normalized_signed_marginal_l1_error": normalized_l1,
        "distribution_solver_status": int(first.status),
        "distribution_solver_message": str(first.message),
        "score_solver_status": int(second.status),
        "score_solver_message": str(second.message),
        "per_start_minimum": None,
        "per_start_maximum": None,
    }
    return selected.astype(np.int64), diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--score-field", default="trace_score")
    parser.add_argument("--eligible-field", default="command_motion_gate_passed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol, protocol_path, protocol_sha = load_protocol(args.protocol)
    summaries = [json.loads(line) for line in Path(args.summaries).read_text(encoding="utf-8").splitlines() if line]
    scores = np.asarray([float(row[args.score_field]) for row in summaries], dtype=np.float64)
    eligible = np.asarray([index for index, row in enumerate(summaries) if bool(row[args.eligible_field])])
    target = dataset_marginal_targets(
        args.dataset, protocol, protocol["candidate"]["selected_mode_counts"]
    )
    selected, diagnostics = select_v10_trajectories(
        summaries, scores, eligible, protocol=protocol, marginal_targets=target
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_sha,
                "selected_indices": selected.tolist(),
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
