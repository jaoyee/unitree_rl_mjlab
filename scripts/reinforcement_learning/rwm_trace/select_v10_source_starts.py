"""Select the exact 1,024 unique source states required by TRACE V10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_trace.select_v10_trajectories import dataset_marginal_targets
from scripts.reinforcement_learning.rwm_trace.v10_protocol import (
    COMMAND_MODES,
    MODE_TO_ID,
    classify_command,
    load_protocol,
    marginal_vector,
    sha256_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol, protocol_path, protocol_sha = load_protocol(args.protocol)
    candidate = protocol["candidate"]
    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    commands_grid = stack_time_key(dataset, "commands").reshape(*stack_time_key(dataset, "states").shape[:2], 3)
    episodes_grid = stack_time_key(dataset, "episode_ids").reshape(commands_grid.shape[:2]).long()
    timesteps_grid = stack_time_key(dataset, "timesteps").reshape(commands_grid.shape[:2]).long()
    history = int(candidate["minimum_source_timestep"])
    eligible = timesteps_grid >= history
    current_start = history - 1
    eligible[:current_start] = False
    for offset in range(1, history):
        eligible[current_start:] &= (
            episodes_grid[current_start:] == episodes_grid[current_start - offset : -offset]
        )
        eligible[current_start:] &= (
            timesteps_grid[current_start:]
            == timesteps_grid[current_start - offset : -offset] + offset
        )
    eligible_ids = torch.nonzero(eligible.reshape(-1), as_tuple=False).flatten().numpy()
    commands = commands_grid.reshape(-1, 3).detach().cpu().numpy()[eligible_ids]
    modes = np.asarray([MODE_TO_ID[classify_command(command)] for command in commands], dtype=np.int8)
    marginals = np.stack([marginal_vector(command, protocol) for command in commands])
    mode_targets = {mode: int(candidate["source_mode_counts"][mode]) for mode in COMMAND_MODES}
    marginal_targets = dataset_marginal_targets(dataset_path, protocol, mode_targets)
    availability = {mode: int(np.sum(modes == MODE_TO_ID[mode])) for mode in COMMAND_MODES}
    deficits = {
        mode: {"available": availability[mode], "required": mode_targets[mode]}
        for mode in COMMAND_MODES
        if availability[mode] < mode_targets[mode]
    }
    if deficits:
        raise ValueError(f"Dataset cannot provide V10 source mode quotas: {deficits}")

    n = len(eligible_ids)
    slack_offset = n
    variable_count = n + 36
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for mode in COMMAND_MODES:
        indices = np.flatnonzero(modes == MODE_TO_ID[mode])
        rows.extend([row] * len(indices)); cols.extend(indices.tolist()); values.extend([1.0] * len(indices))
        lower.append(float(mode_targets[mode])); upper.append(float(mode_targets[mode])); row += 1
    for marginal_id in range(18):
        indices = np.flatnonzero(marginals[:, marginal_id])
        rows.extend([row] * len(indices)); cols.extend(indices.tolist()); values.extend([1.0] * len(indices))
        rows.extend((row, row)); cols.extend((slack_offset + marginal_id, slack_offset + 18 + marginal_id))
        values.extend((-1.0, 1.0))
        lower.append(float(marginal_targets[marginal_id])); upper.append(float(marginal_targets[marginal_id])); row += 1
    matrix = coo_matrix((values, (rows, cols)), shape=(row, variable_count)).tocsr()
    integrality = np.concatenate((np.ones(n, dtype=np.int8), np.zeros(36, dtype=np.int8)))
    bounds = Bounds(np.zeros(variable_count), np.concatenate((np.ones(n), np.full(36, np.inf))))
    options = {
        "time_limit": float(candidate["milp_time_limit_seconds"]),
        "mip_rel_gap": float(candidate["milp_relative_gap"]),
    }
    distribution_objective = np.zeros(variable_count)
    weights = 1.0 / np.maximum(marginal_targets, 1.0)
    distribution_objective[slack_offset : slack_offset + 18] = weights
    distribution_objective[slack_offset + 18 :] = weights
    first = milp(
        c=distribution_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options=options,
    )
    if first.x is None:
        raise RuntimeError(f"V10 source distribution MILP failed: {first.message}")
    optimum = float(distribution_objective @ first.x)
    rng = np.random.default_rng(int(args.seed))
    priority = rng.random(n)
    second_objective = np.zeros(variable_count)
    second_objective[:n] = -priority
    second_matrix = vstack((matrix, csr_matrix(distribution_objective.reshape(1, -1))), format="csr")
    second = milp(
        c=second_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            second_matrix,
            np.concatenate((np.asarray(lower), [-np.inf])),
            np.concatenate((np.asarray(upper), [optimum + 1.0e-6])),
        ),
        options=options,
    )
    if second.x is None:
        raise RuntimeError(f"V10 source priority MILP failed: {second.message}")
    local = np.flatnonzero(second.x[:n] > 0.5)
    source_ids = eligible_ids[local]
    if len(source_ids) != int(candidate["source_start_count"]) or len(np.unique(source_ids)) != len(source_ids):
        raise RuntimeError("V10 source selector returned the wrong number of unique start states.")
    actual_modes = {mode: int(np.sum(modes[local] == MODE_TO_ID[mode])) for mode in COMMAND_MODES}
    actual_marginals = marginals[local].sum(axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_ids": torch.as_tensor(source_ids, dtype=torch.long),
            "metadata": {
                "schema": "go2_trace_v10_source_starts_v1",
                "dataset": str(dataset_path),
                "dataset_sha256": sha256_path(dataset_path),
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_sha,
                "seed": int(args.seed),
                "mode_counts": actual_modes,
                "signed_marginal_counts": actual_marginals.astype(int).tolist(),
            },
        },
        output,
    )
    batch_paths: list[str] = []
    if args.batch_output_dir:
        if args.batch_size <= 0 or len(source_ids) % args.batch_size != 0:
            raise ValueError("V10 source batch size must divide the exact source count.")
        batch_root = Path(args.batch_output_dir).expanduser().resolve()
        batch_root.mkdir(parents=True, exist_ok=True)
        for batch_id, start in enumerate(range(0, len(source_ids), args.batch_size)):
            batch_ids = source_ids[start : start + args.batch_size]
            batch_path = batch_root / f"source_ids_batch_{batch_id:02d}.pt"
            torch.save(
                {
                    "source_ids": torch.as_tensor(batch_ids, dtype=torch.long),
                    "metadata": {
                        "schema": "go2_trace_v10_source_start_batch_v1",
                        "parent": str(output.resolve()),
                        "batch_id": batch_id,
                        "batch_size": len(batch_ids),
                        "protocol_sha256": protocol_sha,
                        "dataset_sha256": sha256_path(dataset_path),
                    },
                },
                batch_path,
            )
            batch_paths.append(str(batch_path))
    report = {
        "schema": "go2_trace_v10_source_start_report_v1",
        "source_count": int(len(source_ids)),
        "eligible_source_count": int(len(eligible_ids)),
        "mode_counts": actual_modes,
        "target_mode_counts": mode_targets,
        "signed_marginal_counts": actual_marginals.astype(int).tolist(),
        "target_signed_marginal_counts": marginal_targets.astype(int).tolist(),
        "normalized_signed_marginal_l1_error": float(
            np.abs(actual_marginals - marginal_targets).sum() / max(marginal_targets.sum(), 1)
        ),
        "distribution_solver_status": int(first.status),
        "priority_solver_status": int(second.status),
        "batch_paths": batch_paths,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
