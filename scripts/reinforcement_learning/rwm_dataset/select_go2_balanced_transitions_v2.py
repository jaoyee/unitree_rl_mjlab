"""Exact eight-mode selector with shared signed-bin targets and continuity preference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from select_go2_matched_blocks_v2 import filter_dataset, time_env
from select_go2_matched_intervals_v2 import MODE_NAMES, TARGETS, labels_and_marginals, sha256
from select_go2_matched_sequence_milp_v2 import build_segments, flatten_env_major


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--marginal-target-json", required=True)
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument("--expert-collector-id", type=int, default=1)
    parser.add_argument("--zero-epsilon", type=float, default=1e-3)
    parser.add_argument("--x-edges", type=float, nargs=4, default=(0.05, 0.20, 0.35, 0.50))
    parser.add_argument("--y-edges", type=float, nargs=4, default=(0.03, 0.087, 0.143, 0.20))
    parser.add_argument("--yaw-edges", type=float, nargs=4, default=(0.05, 0.167, 0.283, 0.40))
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--continuity-weight", type=float, default=2e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    target_path = Path(args.marginal_target_json).expanduser()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    commands_te = time_env(data["commands"], 3)
    modes_te, marginals_te = labels_and_marginals(commands_te, args)
    modes = np.swapaxes(modes_te, 0, 1).reshape(-1)
    marginals = np.swapaxes(marginals_te, 0, 1).reshape(-1, 18)
    terminations = flatten_env_major(time_env(data["terminations"])).reshape(-1)
    episodes = flatten_env_major(time_env(data["episode_ids"])).reshape(-1)
    timesteps = flatten_env_major(time_env(data["timesteps"])).reshape(-1)
    allowed = (modes >= 0) & (terminations <= 0.5)
    if args.expert_only:
        collectors = flatten_env_major(time_env(data["collector_types"])).reshape(-1)
        allowed &= collectors == args.expert_collector_id
    target = np.asarray(json.loads(target_path.read_text())["marginal_target_counts"], dtype=float)
    n = len(modes)
    aux_offset = n
    variables = n + 36
    rows, cols, values, lower, upper = [], [], [], [], []
    row = 0
    for mode_id, count in enumerate(TARGETS):
        indices = np.flatnonzero(modes == mode_id)
        rows.extend([row] * len(indices)); cols.extend(indices.tolist()); values.extend([1.0] * len(indices))
        lower.append(float(count)); upper.append(float(count)); row += 1
    for marginal_id in range(18):
        indices = np.flatnonzero(marginals[:, marginal_id])
        rows.extend([row] * len(indices)); cols.extend(indices.tolist()); values.extend([1.0] * len(indices))
        rows.extend((row, row)); cols.extend((aux_offset + marginal_id, aux_offset + 18 + marginal_id)); values.extend((-1.0, 1.0))
        lower.append(float(target[marginal_id])); upper.append(float(target[marginal_id])); row += 1
    matrix = coo_matrix((values, (rows, cols)), shape=(row, variables)).tocsr()
    continuity = np.zeros(n, dtype=float)
    segments = build_segments(allowed, episodes, timesteps)
    for start, end in segments:
        positions = np.arange(end - start)
        continuity[start:end] = np.minimum(positions + 1, end - start - positions)
    objective = np.zeros(variables)
    objective[:n] = -float(args.continuity_weight) * np.minimum(continuity, 200.0)
    weights = 1.0 / np.maximum(target, 1.0)
    objective[aux_offset:aux_offset + 18] = weights
    objective[aux_offset + 18:] = weights
    result = milp(
        c=objective,
        integrality=np.concatenate((np.ones(n), np.zeros(36))),
        bounds=Bounds(
            np.zeros(variables),
            np.concatenate((allowed.astype(float), np.full(36, np.inf))),
        ),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": args.time_limit, "mip_rel_gap": 0.0},
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
    time_steps, num_envs = commands_te.shape[:2]
    blocks = [[(flat % time_steps, flat // time_steps) for flat in block] for block in blocks_flat]
    output = filter_dataset(data, blocks, (time_steps, num_envs))
    selected_ids = np.flatnonzero(selected)
    actual_modes = np.bincount(modes[selected_ids], minlength=8)
    actual_marginals = marginals[selected_ids].sum(0)
    metadata = dict(output.get("metadata") or {})
    metadata.update({
        "condition_id": args.condition_id,
        "selection_kind": "exact_mode_shared_marginal_continuity_preferred_v2",
        "selection_source": str(input_path.resolve()),
        "selection_source_sha256": sha256(input_path),
        "selection_marginal_target": str(target_path.resolve()),
        "selection_marginal_target_sha256": sha256(target_path),
        "selection_mode_counts": dict(zip(MODE_NAMES, map(int, actual_modes), strict=True)),
        "selection_marginal_counts": actual_marginals.tolist(),
        "selection_expert_only": bool(args.expert_only),
        "selection_continuity_weight": float(args.continuity_weight),
        "num_time_steps": 25000, "num_transitions": 25000, "num_episodes": len(blocks),
    })
    output["metadata"] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    run_lengths = list(map(len, blocks))
    report = {
        "schema": "go2_balanced_transition_selection_v2", "condition_id": args.condition_id,
        "input": str(input_path.resolve()), "output": str(output_path.resolve()),
        "selected_transitions": len(selected_ids), "selected_runs": len(blocks),
        "run_length_min": min(run_lengths), "run_length_max": max(run_lengths),
        "valid_40_step_starts": sum(max(0, length - 39) for length in run_lengths),
        "solver_status": int(result.status), "solver_message": result.message,
        "actual_mode_counts": actual_modes.tolist(),
        "actual_marginal_counts": actual_marginals.tolist(),
        "target_marginal_counts": target.tolist(),
        "normalized_marginal_l1_error": float(np.abs(actual_marginals - target).sum() / target.sum()),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "condition_id", "selected_transitions", "selected_runs", "valid_40_step_starts",
        "normalized_marginal_l1_error", "solver_status", "solver_message",
    )}, indent=2))


if __name__ == "__main__":
    main()
