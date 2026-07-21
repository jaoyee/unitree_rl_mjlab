#!/usr/bin/env python3
"""Build one exact, distribution-constrained Go2 TRACE V10 replay shard."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# This script performs thousands of small CPU tensor reductions. Large server
# defaults (hundreds of OpenMP threads) make that workload substantially slower.
torch.set_num_threads(min(8, torch.get_num_threads()))

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.reference_summary import (
    attach_references_to_summaries,
    load_reference_map,
)
from scripts.reinforcement_learning.rwm_trace.scorer import load_scorer_checkpoint, score_summaries
from scripts.reinforcement_learning.rwm_trace.select_v10_trajectories import (
    dataset_marginal_targets,
    dataset_mode_bucket_probabilities,
    select_v10_trajectories,
)
from scripts.reinforcement_learning.rwm_trace.trajectory import summarize_go2_trajectory
from scripts.reinforcement_learning.rwm_trace.v10_metric_selection import (
    load_metric_config,
    select_global_metric_pilot,
)
from scripts.reinforcement_learning.rwm_trace.v10_protocol import (
    MODE_TO_ID,
    load_protocol,
    signed_magnitude_buckets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate_dataset", required=True)
    parser.add_argument("--target_dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--refresh_cycle", type=int, required=True)
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer_checkpoint", default=None)
    parser.add_argument("--selection", choices=("scorer", "metric", "random", "all"), default="scorer")
    parser.add_argument(
        "--metric_pilot_config",
        default=None,
        help="Enable scorer-free unconstrained global metric/random Top-25%% selection.",
    )
    parser.add_argument("--select_ratio", type=float, default=0.25)
    parser.add_argument(
        "--per_start_target_count",
        type=int,
        default=0,
        help=(
            "Fixed number selected for each dataset start when selection_scope=per_start. "
            "Use this when extra rollout branches are appended only to repair capacity."
        ),
    )
    parser.add_argument(
        "--selection_scope",
        choices=("global", "per_start"),
        default="global",
        help="Select globally or independently among rollouts sharing one dataset start state.",
    )
    parser.add_argument(
        "--command_motion_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require non-stand trajectories to realize the commanded linear/yaw motion before scoring.",
    )
    parser.add_argument("--min_linear_realization_ratio", type=float, default=0.1)
    parser.add_argument("--min_yaw_realization_ratio", type=float, default=0.1)
    parser.add_argument(
        "--min_linear_net_displacement",
        type=float,
        default=1.0e-3,
        help="Reject active linear commands whose projected displacement is only numerical noise.",
    )
    parser.add_argument(
        "--min_yaw_net_change",
        type=float,
        default=1.0e-3,
        help="Reject active yaw commands whose integrated yaw response is only numerical noise.",
    )
    parser.add_argument("--max_command_direction_violation_rate", type=float, default=0.5)
    parser.add_argument(
        "--max_hard_direction_violation_rate",
        type=float,
        default=0.5,
        help="Only candidates above this direction-error rate are permanently ineligible.",
    )
    parser.add_argument(
        "--command_mode_weights",
        default=None,
        help=(
            "Optional soft selection quotas such as stand:0.08,pure_x:0.25,... . "
            "Unavailable non-stand quota is redistributed; stand is always capped at its configured fraction."
        ),
    )
    parser.add_argument("--trajectory_length", type=int, default=100)
    parser.add_argument("--trajectory_stride", type=int, default=100)
    parser.add_argument(
        "--include_terminal_prefixes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include variable-length episode prefixes that end at an environment termination.",
    )
    parser.add_argument("--minimum_terminal_length", type=int, default=20)
    parser.add_argument(
        "--failure_trajectory_ratio",
        type=float,
        default=0.0,
        help="Minimum fraction of selected trajectories reserved for terminal/failure trajectories.",
    )
    parser.add_argument(
        "--failure_transition_ratio",
        type=float,
        default=None,
        help=(
            "Target fraction of replay rows originating from terminal trajectories. "
            "When set, this supersedes failure_trajectory_ratio and compensates for shorter failures."
        ),
    )
    parser.add_argument(
        "--terminal_penalty",
        type=float,
        default=-10.0,
        help="Legacy terminal reward used only when --override-terminal-reward is enabled.",
    )
    parser.add_argument(
        "--override_terminal_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy shaping option. Disabled by default to match frozen-RWM rewards exactly.",
    )
    parser.add_argument("--failure_backprop_steps", type=int, default=0)
    parser.add_argument(
        "--failure_backprop_penalty",
        type=float,
        default=0.0,
        help="Maximum positive cost subtracted immediately before termination.",
    )
    parser.add_argument("--action_saturation_threshold", type=float, default=0.95)
    parser.add_argument(
        "--max_saturation_fraction",
        type=float,
        default=0.5,
        help="Reject trajectories exceeding this fraction of saturated transitions.",
    )
    parser.add_argument(
        "--max_reset_reconstruction_error",
        type=float,
        default=5.0e-3,
        help="Reject candidates whose recorded reset reconstruction error exceeds this bound.",
    )
    parser.add_argument("--action_saturation_penalty_scale", type=float, default=0.0)
    parser.add_argument("--action_delta_penalty_scale", type=float, default=0.0)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_source", choices=("rwm_aligned", "dataset"), default="rwm_aligned")
    parser.add_argument("--reward_version", choices=("v1", "v2_hierarchical"), default="v1")
    parser.add_argument("--reward_command_response_weight", type=float, default=2.0)
    parser.add_argument("--reward_yaw_command_response_weight", type=float, default=1.0)
    parser.add_argument("--reward_wrong_direction_weight", type=float, default=-2.0)
    parser.add_argument("--reward_response_shortfall_weight", type=float, default=-6.0)
    parser.add_argument("--reward_response_floor", type=float, default=0.30)
    parser.add_argument("--reward_active_command_bias", type=float, default=-0.20)
    parser.add_argument("--reward_uncertainty_penalty_weight", type=float, default=-1.0)
    parser.add_argument("--reward_action_rate_l2", type=float, default=-0.05)
    parser.add_argument("--reward_dof_acc_l2", type=float, default=-2.5e-7)
    parser.add_argument("--reward_dof_torques_l2", type=float, default=-2.5e-5)
    parser.add_argument("--reward_command_active_threshold", type=float, default=0.02)
    parser.add_argument("--reward_motion_gate_low", type=float, default=0.05)
    parser.add_argument("--reward_motion_gate_high", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_jsonl", default=None)
    parser.add_argument(
        "--reference_summaries",
        default=None,
        help="JSONL same-start frozen-RWM summaries used to score simulator candidates by residual value.",
    )
    parser.add_argument("--summaries_only", action="store_true")
    parser.add_argument(
        "--allow_legacy_candidates",
        action="store_true",
        help="Explicitly bypass V5 controlled-reset gates for diagnostic reproduction only.",
    )
    return parser.parse_args()


def _make_policy_obs(state: torch.Tensor, command: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    return torch.cat([state[..., :33], command, previous_action], dim=-1).float()


def _candidate_windows(
    dataset: dict[str, Any],
    length: int,
    stride: int,
    *,
    include_terminal_prefixes: bool,
    minimum_terminal_length: int,
) -> list[dict[str, Any]]:
    tensors = {
        key: stack_time_key(dataset, key)
        for key in ("states", "actions", "next_states", "contacts", "terminations", "commands", "rewards")
    }
    optional_diagnostics = {
        key: stack_time_key(dataset, key)
        for key in (
            "trace_base_positions_w",
            "trace_foot_positions_w",
            "trace_foot_velocities_w",
        )
        if dataset.get(key) is not None
    }
    step_dt = float((dataset.get("metadata") or {}).get("dt", 0.02))
    if dataset.get("prev_actions") is None:
        raise ValueError(
            "TRACE candidate dataset is missing prev_actions. Refusing the unsafe global-shift fallback."
        )
    previous_actions = stack_time_key(dataset, "prev_actions")
    episode_ids = stack_time_key(dataset, "episode_ids") if dataset.get("episode_ids") is not None else None
    dones = stack_time_key(dataset, "dones") if dataset.get("dones") is not None else None
    timeouts = stack_time_key(dataset, "timeouts") if dataset.get("timeouts") is not None else None
    trace_start_state_ids = dataset.get("trace_start_state_ids")
    if trace_start_state_ids is not None:
        trace_start_state_ids = torch.as_tensor(trace_start_state_ids).reshape(-1)
    reset_errors = (
        stack_time_key(dataset, "trace_reset_reconstruction_errors")
        if dataset.get("trace_reset_reconstruction_errors") is not None
        else None
    )
    valid_masks = (
        stack_time_key(dataset, "trace_valid_masks").bool()
        if dataset.get("trace_valid_masks") is not None
        else torch.ones(tensors["states"].shape[:2], dtype=torch.bool)
    )
    mismatch = {
        "target_gap": dict((dataset.get("metadata") or {}).get("target_gap") or {}),
        "env_randomization": dict((dataset.get("metadata") or {}).get("env_randomization") or {}),
    }
    time_steps, num_envs = tensors["states"].shape[:2]
    candidates: list[dict[str, Any]] = []

    def append_candidate(env_id: int, start: int, stop: int) -> bool:
        trajectory_length = stop - start
        if not bool(valid_masks[start:stop, env_id].all()):
            return False
        unsafe_term = tensors["terminations"][start:stop, env_id].bool().reshape(trajectory_length, -1).any(dim=-1)
        if dones is not None:
            env_done = dones[start:stop, env_id].bool().reshape(trajectory_length, -1).any(dim=-1)
            if timeouts is not None:
                env_timeout = timeouts[start:stop, env_id].bool().reshape(trajectory_length, -1).any(dim=-1)
                env_done = env_done & ~env_timeout
            term = unsafe_term | env_done
        else:
            term = unsafe_term
        if term[:-1].any():
            return False
        trajectory = {key: value[start:stop, env_id] for key, value in tensors.items()}
        trajectory["terminations"] = term
        trajectory["prev_actions"] = previous_actions[start:stop, env_id]
        trajectory["trajectory_id"] = f"env{env_id:05d}_start{start:08d}_len{trajectory_length:04d}"
        trajectory["start_state_id"] = (
            int(trace_start_state_ids[env_id])
            if trace_start_state_ids is not None
            else env_id * time_steps + start
        )
        trajectory["reset_reconstruction_error"] = (
            float(reset_errors[start:stop, env_id].float().mean())
            if reset_errors is not None
            else float("nan")
        )
        trajectory["simulator_mismatch"] = mismatch
        trajectory["step_dt"] = step_dt
        for key, value in optional_diagnostics.items():
            trajectory[key] = value[start:stop, env_id]
        candidates.append(trajectory)
        return True

    for env_id in range(num_envs):
        if episode_ids is None:
            segments = [(0, time_steps)]
        else:
            ids = episode_ids[:, env_id].reshape(time_steps, -1)[:, 0]
            boundaries = (ids[1:] != ids[:-1]).nonzero(as_tuple=False).flatten().add(1).tolist()
            points = [0, *boundaries, time_steps]
            segments = list(zip(points[:-1], points[1:]))

        # Candidate collection keeps a rectangular tensor, but after the first
        # terminal transition every later row for that env is explicitly
        # invalid.  Split those rows out before constructing windows.
        valid = valid_masks[:, env_id].reshape(-1)
        valid_boundaries = ((~valid[1:]) & valid[:-1]).nonzero(as_tuple=False).flatten().add(1).tolist()
        if valid_boundaries:
            valid_stop = int(valid_boundaries[0])
            segments = [
                (start, min(stop, valid_stop))
                for start, stop in segments
                if start < valid_stop and min(stop, valid_stop) > start
            ]

        emitted: set[tuple[int, int]] = set()
        for segment_start, segment_stop in segments:
            segment_length = segment_stop - segment_start
            for start in range(segment_start, segment_stop - length + 1, stride):
                stop = start + length
                if append_candidate(env_id, start, stop):
                    emitted.add((start, stop))

            if not include_terminal_prefixes or segment_length < minimum_terminal_length:
                continue
            unsafe_term = tensors["terminations"][segment_start:segment_stop, env_id].bool().reshape(
                segment_length, -1
            ).any(dim=-1)
            if dones is not None:
                env_done = dones[segment_start:segment_stop, env_id].bool().reshape(
                    segment_length, -1
                ).any(dim=-1)
                if timeouts is not None:
                    env_timeout = timeouts[segment_start:segment_stop, env_id].bool().reshape(
                        segment_length, -1
                    ).any(dim=-1)
                    env_done = env_done & ~env_timeout
                terminal = unsafe_term | env_done
            else:
                terminal = unsafe_term
            terminal_indices = terminal.nonzero(as_tuple=False).flatten()
            if terminal_indices.numel() == 0:
                continue
            stop = segment_start + int(terminal_indices[0]) + 1
            start = max(segment_start, stop - length)
            if stop - start >= minimum_terminal_length and (start, stop) not in emitted:
                append_candidate(env_id, start, stop)
    return candidates


def _validate_controlled_candidates(dataset: dict[str, Any], *, allow_legacy: bool) -> None:
    metadata = dict(dataset.get("metadata") or {})
    trace = dict(metadata.get("trace_candidates") or {})
    errors: list[str] = []
    if not trace.get("enabled", False):
        errors.append("metadata.trace_candidates.enabled is not true")
    if not trace.get("controlled_branch_domain", False):
        errors.append("branch domain parameters were not synchronized")
    if not trace.get("command_from_source_transition", False):
        errors.append("proposal command was not restored from the source transition")
    if not trace.get("stop_on_done", False):
        errors.append("collector did not preserve and stop at the terminal boundary")
    if dataset.get("trace_valid_masks") is None:
        errors.append("trace_valid_masks is missing")
    if dataset.get("prev_actions") is None:
        errors.append("prev_actions is missing")
    if dataset.get("trace_start_state_ids") is None:
        errors.append("trace_start_state_ids is missing")
    reset_mode = trace.get("reset_mode")
    if reset_mode not in {"exact_snapshot", "canonical_real_projection"}:
        errors.append(f"unsupported reset_mode={reset_mode!r}")
    repeated_identity = dict(trace.get("one_step_identity") or {})
    if not repeated_identity.get("passed", False):
        errors.append("repeated snapshot/action identity gate did not pass")
    source_identity = dict(trace.get("source_transition_identity") or {})
    if reset_mode == "exact_snapshot":
        if not trace.get("reset_is_exact", False):
            errors.append("exact_snapshot candidate does not declare reset_is_exact=true")
        source_identity_required = source_identity.get("required", True)
        if source_identity_required:
            if not source_identity.get("available", False) or not source_identity.get("passed", False):
                errors.append("exact snapshot did not reproduce the dataset source next_state")
        elif not source_identity.get("available", False):
            errors.append("imperfect-simulator source-transition comparison is unavailable")
        elif not source_identity.get("mismatch_expected", False):
            errors.append("disabled source-transition identity does not declare mismatch_expected=true")
    elif reset_mode == "canonical_real_projection":
        if trace.get("reset_is_exact", False):
            errors.append("canonical real projection incorrectly claims an exact reset")
        if source_identity.get("available", False):
            errors.append("canonical real projection incorrectly claims source-transition identity")
    if errors and not allow_legacy:
        raise ValueError(
            "Candidate dataset failed controlled TRACE gates: " + "; ".join(errors)
        )

    if errors:
        return
    states = stack_time_key(dataset, "states")
    commands = stack_time_key(dataset, "commands")
    valid = stack_time_key(dataset, "trace_valid_masks").bool()
    start_ids = torch.as_tensor(dataset["trace_start_state_ids"]).reshape(-1)
    for start_id in torch.unique(start_ids):
        env_ids = (start_ids == start_id).nonzero(as_tuple=False).flatten()
        first_rows = []
        for env_id in env_ids.tolist():
            row_ids = valid[:, env_id].nonzero(as_tuple=False).flatten()
            if row_ids.numel() == 0:
                continue
            first_rows.append((states[int(row_ids[0]), env_id], commands[int(row_ids[0]), env_id]))
        if len(first_rows) < 2:
            continue
        state_ref, command_ref = first_rows[0]
        if any(not torch.allclose(state_ref, state, atol=1.0e-5, rtol=1.0e-5) for state, _ in first_rows[1:]):
            raise ValueError(f"start_state_id={int(start_id)} branches do not share the same realized state")
        if any(not torch.allclose(command_ref, command, atol=1.0e-7, rtol=0.0) for _, command in first_rows[1:]):
            raise ValueError(f"start_state_id={int(start_id)} branches do not share the same command")


def _select_indices(
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(summaries)
    if count == 0:
        raise ValueError("Candidate dataset contains no complete trajectory windows.")
    safety_eligible_indices = np.asarray(
        [
            index
            for index, summary in enumerate(summaries)
            if not bool(summary.get("nonfinite_flag", True))
            and np.isfinite(float(summary.get("reset_reconstruction_error", float("nan"))))
            and float(summary["reset_reconstruction_error"])
            <= float(args.max_reset_reconstruction_error)
            and float(summary.get("action_saturation_fraction", float("inf")))
            <= float(args.max_saturation_fraction)
        ],
        dtype=np.int64,
    )
    if len(safety_eligible_indices) == 0:
        raise ValueError("Safety filter rejected every candidate trajectory.")
    motion_rejections: dict[str, int] = {}
    strict_eligible: list[int] = []
    for index in safety_eligible_indices:
        summary = summaries[int(index)]
        reason = _motion_gate_rejection_reason(summary, args)
        if reason is None:
            strict_eligible.append(int(index))
        else:
            motion_rejections[reason] = motion_rejections.get(reason, 0) + 1
    eligible_indices = np.asarray(strict_eligible, dtype=np.int64)
    if len(eligible_indices) == 0:
        raise ValueError(f"Command-motion gate rejected every candidate trajectory: {motion_rejections}")
    strict_set = set(strict_eligible)
    diagnostics = {
        "strict_eligible_count": len(strict_eligible),
        "fallback_eligible_count": 0,
        "hard_rejection_count": len(safety_eligible_indices) - len(eligible_indices),
        "rejection_counts": motion_rejections,
    }
    setattr(args, "_selection_diagnostics", diagnostics)
    metric_config = getattr(args, "_metric_pilot_config", None)
    if metric_config is not None:
        selected, scores, metric_diagnostics = select_global_metric_pilot(
            summaries,
            eligible_indices,
            config=metric_config,
            selection=str(args.selection),
            seed=int(args.seed),
        )
        diagnostics.update(metric_diagnostics)
        diagnostics["target_selected_count"] = int(metric_config["selection"]["selected_trajectory_count"])
        diagnostics["selected_strict_count"] = int(len(selected))
        diagnostics["selected_fallback_count"] = 0
        setattr(args, "_selection_diagnostics", diagnostics)
        return selected, scores
    if args.selection == "all":
        scores = np.zeros(count, dtype=np.float32)
        return eligible_indices, scores
    rng = np.random.default_rng(args.seed)
    if args.selection == "random":
        scores = rng.random(count, dtype=np.float32)
    else:
        if not args.scorer_checkpoint:
            raise ValueError("--scorer_checkpoint is required for scorer selection.")
        model, stats, _ = load_scorer_checkpoint(args.scorer_checkpoint)
        scores = score_summaries(model, summaries, stats)
    jitter = rng.uniform(0.0, 1.0e-9, size=count)
    if getattr(args, "protocol", None):
        protocol, _, _ = load_protocol(args.protocol)
        marginal_targets = dataset_marginal_targets(
            args.target_dataset,
            protocol,
            protocol["candidate"]["selected_mode_counts"],
        )
        selected, v10_diagnostics = select_v10_trajectories(
            summaries,
            scores,
            eligible_indices,
            protocol=protocol,
            marginal_targets=marginal_targets,
        )
        diagnostics.update(v10_diagnostics)
        diagnostics["target_selected_count"] = int(protocol["candidate"]["selected_trajectory_count"])
        setattr(args, "_selection_diagnostics", diagnostics)
        return selected, scores

    failure_indices = np.asarray(
        [
            int(index)
            for index in eligible_indices
            if bool(summaries[int(index)].get("terminal_flag", False))
        ],
        dtype=np.int64,
    )
    transition_target = getattr(args, "failure_transition_ratio", None)
    trajectory_minimum = float(args.failure_trajectory_ratio)
    quota_disabled = (
        (transition_target is None and trajectory_minimum <= 0.0)
        or (transition_target is not None and float(transition_target) <= 0.0)
    )
    ranking_scores = scores + jitter

    def ranked_with_strict_priority(indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            sorted(
                map(int, indices),
                key=lambda index: (index not in strict_set, -float(ranking_scores[index])),
            ),
            dtype=np.int64,
        )

    ranked = ranked_with_strict_priority(eligible_indices)
    if quota_disabled:
        # A zero target disables forced failure composition. Natural terminal
        # trajectories remain eligible and are selected solely by the chosen
        # scorer/random ranking.
        if getattr(args, "selection_scope", "global") == "per_start":
            grouped: dict[str, list[int]] = {}
            safety_group_sizes: dict[str, int] = {}
            for index in safety_eligible_indices:
                key = str(summaries[int(index)].get("comparison_group_key", ""))
                if not key:
                    raise ValueError("per_start selection requires comparison_group_key on every trajectory.")
                safety_group_sizes[key] = safety_group_sizes.get(key, 0) + 1
            for index in eligible_indices:
                key = str(summaries[int(index)].get("comparison_group_key", ""))
                grouped.setdefault(key, []).append(int(index))
            selected = []
            missing_groups = sorted(set(safety_group_sizes) - set(grouped))
            fixed_group_target = int(getattr(args, "per_start_target_count", 0))
            required_by_group = {
                key: (
                    fixed_group_target
                    if fixed_group_target > 0
                    else max(1, int(math.ceil(size * float(args.select_ratio))))
                )
                for key, size in safety_group_sizes.items()
            }
            requested_selected_count = sum(required_by_group.values())
            insufficient_groups = {
                key: {
                    "eligible": len(grouped.get(key, ())),
                    "required": required_by_group[key],
                }
                for key in safety_group_sizes
                if len(grouped.get(key, ())) < required_by_group[key]
            }
            if insufficient_groups:
                preview = dict(list(sorted(insufficient_groups.items()))[:20])
                raise ValueError(
                    "Motion-valid candidates cannot satisfy per-start top-k selection. "
                    "Regenerate candidates with more rollout branches; replay will not be underfilled. "
                    f"affected_groups={len(insufficient_groups)}, preview={preview}"
                )
            for key in sorted(safety_group_sizes):
                group = np.asarray(grouped[key], dtype=np.int64)
                group_count = required_by_group[key]
                group_ranked = ranked_with_strict_priority(group)
                selected.extend(map(int, group_ranked[: min(group_count, len(group_ranked))]))
            selected_array = np.asarray(selected, dtype=np.int64)
            diagnostics["requested_selected_count"] = requested_selected_count
            diagnostics["target_selected_count"] = len(selected_array)
            diagnostics["selection_shortfall_count"] = 0
            diagnostics["start_group_count"] = len(safety_group_sizes)
            diagnostics["missing_start_group_count"] = len(missing_groups)
            diagnostics["missing_start_group_keys"] = missing_groups[:20]
            diagnostics["additional_branch_fill_count"] = 0
            diagnostics["selected_strict_count"] = sum(int(index) in strict_set for index in selected_array)
            diagnostics["selected_fallback_count"] = len(selected_array) - diagnostics["selected_strict_count"]
            return selected_array, scores
        selected_count = max(1, int(math.ceil(len(eligible_indices) * float(args.select_ratio))))
        weights = _parse_command_mode_weights(getattr(args, "command_mode_weights", None))
        selected_array = (
            _select_from_mode_pool(eligible_indices, summaries, ranking_scores, strict_set, weights, selected_count)
            if weights
            else ranked[:selected_count].astype(np.int64)
        )
        diagnostics["target_selected_count"] = selected_count
        diagnostics["selected_strict_count"] = sum(int(index) in strict_set for index in selected_array)
        diagnostics["selected_fallback_count"] = len(selected_array) - diagnostics["selected_strict_count"]
        return selected_array, scores
    if getattr(args, "selection_scope", "global") != "global":
        raise ValueError("Forced failure quotas are incompatible with per_start selection.")
    selected_count = max(1, int(math.ceil(len(eligible_indices) * float(args.select_ratio))))
    if transition_target is None:
        failure_count = min(
            len(failure_indices),
            int(math.ceil(selected_count * trajectory_minimum)),
        )
    else:
        target = float(transition_target)
        n_step = int(getattr(args, "n_step", 3))
        success_indices = [int(index) for index in eligible_indices if int(index) not in set(map(int, failure_indices))]

        def row_count(index: int) -> int:
            length = int(summaries[index].get("survival_length", 0))
            return length if bool(summaries[index].get("terminal_flag", False)) else max(0, length - n_step + 1)

        ranked_failures = sorted(map(int, failure_indices), key=lambda index: -(scores[index] + jitter[index]))
        ranked_successes = sorted(success_indices, key=lambda index: -(scores[index] + jitter[index]))
        possibilities = []
        for count in range(0, min(len(ranked_failures), selected_count) + 1):
            success_count = selected_count - count
            if success_count > len(ranked_successes):
                continue
            failure_rows = sum(row_count(index) for index in ranked_failures[:count])
            success_rows = sum(row_count(index) for index in ranked_successes[:success_count])
            ratio = failure_rows / max(failure_rows + success_rows, 1)
            possibilities.append((abs(ratio - target), count))
        if not possibilities:
            raise ValueError("Insufficient terminal/nonterminal candidates for failure transition quota.")
        failure_count = min(possibilities)[1]
    selected_failures = (
        failure_indices[np.argsort(-(scores[failure_indices] + jitter[failure_indices]))[:failure_count]]
        if failure_count
        else np.empty(0, dtype=np.int64)
    )
    # The configured failure ratio is an exact target (subject to candidate
    # availability), not merely a minimum. Excluding only the failures already
    # selected for the quota allowed additional terminal trajectories to leak
    # into the nominal-success portion of the replay.
    all_failure_set = set(map(int, failure_indices))
    selected_successes = [int(index) for index in ranked if int(index) not in all_failure_set][
        : selected_count - failure_count
    ]
    selected = np.asarray([*map(int, selected_failures), *selected_successes], dtype=np.int64)
    return selected.astype(np.int64), scores


def _motion_gate_rejection_reason(summary: dict[str, Any], args: argparse.Namespace) -> str | None:
    if not bool(getattr(args, "command_motion_gate", False)):
        return None
    if str(summary.get("command_mode", "")) == "stand":
        return None
    direction_rate = float(summary.get("command_direction_violation_rate", float("nan")))
    if not np.isfinite(direction_rate) or direction_rate > float(
        getattr(args, "max_hard_direction_violation_rate", 0.9)
    ):
        return "severe_direction_violation"
    if direction_rate > float(getattr(args, "max_command_direction_violation_rate", 0.5)):
        return "direction_violation"
    if bool(summary.get("command_linear_active", False)):
        projected_displacement = float(
            summary.get("command_projected_displacement", float("nan"))
        )
        if not np.isfinite(projected_displacement):
            return "nonfinite_linear_displacement"
        if projected_displacement <= 0.0:
            return "opposite_linear_net_motion"
        if projected_displacement < float(
            getattr(args, "min_linear_net_displacement", 1.0e-3)
        ):
            yaw_change = abs(float(summary.get("yaw_net_change", 0.0)))
            if np.isfinite(yaw_change) and yaw_change > 4.0 * max(
                projected_displacement,
                float(getattr(args, "min_linear_net_displacement", 1.0e-3)),
            ):
                return "linear_command_rotation_only"
            return "near_zero_linear_net_motion"
        ratio = float(summary.get("linear_velocity_realization_ratio_mean", float("nan")))
        if not np.isfinite(ratio) or ratio < float(getattr(args, "min_linear_realization_ratio", 0.2)):
            return "linear_realization"
    if bool(summary.get("command_yaw_active", False)):
        yaw_change = float(summary.get("yaw_net_change", float("nan")))
        expected_yaw_change = float(summary.get("yaw_expected_change", float("nan")))
        if not np.isfinite(yaw_change) or not np.isfinite(expected_yaw_change):
            return "nonfinite_yaw_displacement"
        if yaw_change * expected_yaw_change <= 0.0:
            return "opposite_yaw_net_motion"
        if abs(yaw_change) < float(getattr(args, "min_yaw_net_change", 1.0e-3)):
            return "near_zero_yaw_net_motion"
        ratio = float(summary.get("yaw_velocity_realization_ratio_mean", float("nan")))
        if not np.isfinite(ratio) or ratio < float(getattr(args, "min_yaw_realization_ratio", 0.2)):
            return "yaw_realization"
    return None


def _parse_command_mode_weights(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    weights = {}
    for item in str(spec).split(","):
        name, separator, raw_value = item.strip().partition(":")
        if not separator or not name:
            raise ValueError(f"Bad command mode weight entry: {item!r}")
        value = float(raw_value)
        if value < 0.0:
            raise ValueError(f"Command mode weight must be non-negative: {item!r}")
        weights[name] = value
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Command mode weights must sum to a positive value.")
    return {name: value / total for name, value in weights.items()}


def _mode_target_counts(weights: dict[str, float], target_count: int) -> dict[str, int]:
    raw = {mode: float(weight) * target_count for mode, weight in weights.items()}
    counts = {mode: int(math.floor(value)) for mode, value in raw.items()}
    remainder = target_count - sum(counts.values())
    order = sorted(weights, key=lambda mode: (-(raw[mode] - counts[mode]), mode))
    for mode in order[:remainder]:
        counts[mode] += 1
    return counts


def _select_from_mode_pool(
    eligible: np.ndarray,
    summaries: list[dict[str, Any]],
    ranking_scores: np.ndarray,
    strict_set: set[int],
    weights: dict[str, float],
    target_count: int,
) -> np.ndarray:
    """Allocate from the complete eligible pool and return exactly target_count rows."""

    by_mode: dict[str, list[int]] = {}
    for index in eligible:
        mode = str(summaries[int(index)].get("command_mode", "unknown"))
        by_mode.setdefault(mode, []).append(int(index))
    for values in by_mode.values():
        values.sort(key=lambda index: (index not in strict_set, -float(ranking_scores[index])))

    targets = _mode_target_counts(weights, target_count)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for mode, quota in targets.items():
        for index in by_mode.get(mode, [])[:quota]:
            chosen.append(index)
            chosen_set.add(index)

    # Missing quota is redistributed only among locomotion modes. Stand remains
    # capped at its requested largest-remainder allocation.
    remaining = sorted(
        (
            int(index) for index in eligible
            if int(index) not in chosen_set
            and str(summaries[int(index)].get("command_mode", "unknown")) != "stand"
        ),
        key=lambda index: (index not in strict_set, -float(ranking_scores[index])),
    )
    for index in remaining:
        if len(chosen) >= target_count:
            break
        chosen.append(index)
        chosen_set.add(index)

    if len(chosen) != target_count:
        available = {mode: len(indices) for mode, indices in sorted(by_mode.items())}
        raise ValueError(
            "Cannot satisfy exact command-mode replay size without converting missing locomotion "
            f"quota into stand: selected={len(chosen)}, target={target_count}, available={available}"
        )
    return np.asarray(chosen, dtype=np.int64)


def _validate_per_start_mode_allocation(
    selected: np.ndarray,
    summaries: list[dict[str, Any]],
    weights: dict[str, float],
) -> None:
    target = _mode_target_counts(weights, len(selected))
    actual = {
        mode: sum(str(summaries[int(index)].get("command_mode", "unknown")) == mode for index in selected)
        for mode in weights
    }
    deficits = {
        mode: target[mode] - actual.get(mode, 0)
        for mode in target
        if mode != "stand" and actual.get(mode, 0) < target[mode]
    }
    if actual.get("stand", 0) > target.get("stand", 0):
        raise ValueError(
            "Per-start replay exceeds the stand cap. Regenerate candidates with "
            f"stratified reset-state sampling. target={target}, actual={actual}, deficits={deficits}"
        )


def _fill_per_start_shortfall(
    selected: np.ndarray,
    eligible: np.ndarray,
    summaries: list[dict[str, Any]],
    ranking_scores: np.ndarray,
    strict_set: set[int],
    weights: dict[str, float],
    target_count: int,
) -> np.ndarray:
    if len(selected) > target_count:
        raise RuntimeError(f"Per-start selection produced {len(selected)} rows for target {target_count}.")
    chosen = list(map(int, selected))
    chosen_set = set(chosen)
    targets = _mode_target_counts(weights, target_count) if weights else {}

    def mode(index: int) -> str:
        return str(summaries[index].get("command_mode", "unknown"))

    def counts() -> dict[str, int]:
        return {name: sum(mode(index) == name for index in chosen) for name in targets}

    remaining = [int(index) for index in eligible if int(index) not in chosen_set and mode(int(index)) != "stand"]
    remaining.sort(key=lambda index: (index not in strict_set, -float(ranking_scores[index])))
    while len(chosen) < target_count:
        current = counts()
        deficits = {name for name, target in targets.items() if name != "stand" and current.get(name, 0) < target}
        candidate_position = next(
            (position for position, index in enumerate(remaining) if not deficits or mode(index) in deficits),
            None,
        )
        if candidate_position is None and remaining:
            # The missing mode has no eligible branch. Redistribute only to
            # another locomotion mode; never use stand to hide the shortfall.
            candidate_position = 0
        if candidate_position is None:
            raise ValueError(
                "Cannot fill exact per-start replay size from locomotion candidates: "
                f"selected={len(chosen)}, target={target_count}, mode_counts={current}, targets={targets}"
            )
        index = remaining.pop(candidate_position)
        chosen.append(index)
        chosen_set.add(index)
    return np.asarray(chosen, dtype=np.int64)


def _apply_replay_safety_costs(
    rewards: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    terminated: torch.Tensor,
    *,
    terminal_penalty: float,
    override_terminal_reward: bool = False,
    failure_backprop_steps: int,
    failure_backprop_penalty: float,
    action_saturation_threshold: float,
    action_saturation_penalty_scale: float,
    action_delta_penalty_scale: float,
) -> torch.Tensor:
    rewards = rewards.clone().reshape(-1)
    terminated = terminated.reshape(len(rewards), -1).bool().any(dim=-1)
    action_excess = torch.relu(torch.abs(actions) - float(action_saturation_threshold))
    saturation_cost = torch.mean(torch.square(action_excess), dim=-1)
    action_delta_cost = torch.mean(torch.square(actions - previous_actions), dim=-1)
    rewards -= float(action_saturation_penalty_scale) * saturation_cost
    rewards -= float(action_delta_penalty_scale) * action_delta_cost
    if failure_backprop_steps > 0 and failure_backprop_penalty > 0.0:
        for terminal_index in terminated.nonzero(as_tuple=False).flatten().tolist():
            start = max(0, int(terminal_index) - int(failure_backprop_steps))
            count = int(terminal_index) - start
            if count > 0:
                ramp = torch.linspace(1.0 / count, 1.0, count, dtype=rewards.dtype)
                rewards[start:terminal_index] -= float(failure_backprop_penalty) * ramp
    if override_terminal_reward:
        rewards[terminated] = float(terminal_penalty)
    return rewards


def _align_rewards_with_rwm(
    trajectory: dict[str, Any],
    step_dt: float,
    terminal_penalty: float,
    *,
    override_terminal_reward: bool,
    failure_backprop_steps: int,
    failure_backprop_penalty: float,
    action_saturation_threshold: float,
    action_saturation_penalty_scale: float,
    action_delta_penalty_scale: float,
) -> None:
    """Recompute physical-simulator rewards with the frozen-RWM reward definition."""

    from src.tasks.rwm_velocity.mdp.rewards import (
        Go2RWMRewardState,
        compute_go2_imagination_reward,
    )

    state = Go2RWMRewardState.create(
        num_envs=1,
        action_dim=int(trajectory["actions"].shape[-1]),
        device="cpu",
        step_dt=float(step_dt),
    )
    state.last_joint_vel[0] = trajectory["states"][0, 21:33]
    state.last_action[0] = trajectory["prev_actions"][0]
    rewards: list[torch.Tensor] = []
    for index in range(len(trajectory["states"])):
        reward, _ = compute_go2_imagination_reward(
            state=trajectory["next_states"][index : index + 1],
            action=trajectory["actions"][index : index + 1],
            command=trajectory["commands"][index : index + 1],
            foot_contact=trajectory["contacts"][index : index + 1],
            episode_length=torch.tensor([index], dtype=torch.long),
            reward_state=state,
            epistemic_uncertainty=torch.zeros(1),
        )
        rewards.append(reward[0])
    trajectory["dataset_rewards"] = trajectory["rewards"].clone()
    trajectory["rewards"] = torch.stack(rewards)
    trajectory["rewards"] = _apply_replay_safety_costs(
        trajectory["rewards"],
        trajectory["actions"],
        trajectory["prev_actions"],
        trajectory["terminations"],
        terminal_penalty=terminal_penalty,
        override_terminal_reward=override_terminal_reward,
        failure_backprop_steps=failure_backprop_steps,
        failure_backprop_penalty=failure_backprop_penalty,
        action_saturation_threshold=action_saturation_threshold,
        action_saturation_penalty_scale=action_saturation_penalty_scale,
        action_delta_penalty_scale=action_delta_penalty_scale,
    )


def _align_rewards_with_rwm_batch(
    trajectories: list[dict[str, Any]],
    step_dt: float,
    terminal_penalty: float,
    *,
    override_terminal_reward: bool,
    failure_backprop_steps: int,
    failure_backprop_penalty: float,
    action_saturation_threshold: float,
    action_saturation_penalty_scale: float,
    action_delta_penalty_scale: float,
    reward_version: str = "v1",
    reward_command_response_weight: float = 2.0,
    reward_yaw_command_response_weight: float = 1.0,
    reward_wrong_direction_weight: float = -2.0,
    reward_response_shortfall_weight: float = -6.0,
    reward_response_floor: float = 0.30,
    reward_active_command_bias: float = -0.20,
    reward_uncertainty_penalty_weight: float = -1.0,
    reward_action_rate_l2: float = -0.05,
    reward_dof_acc_l2: float = -2.5e-7,
    reward_dof_torques_l2: float = -2.5e-5,
    reward_command_active_threshold: float = 0.02,
    reward_motion_gate_low: float = 0.05,
    reward_motion_gate_high: float = 0.30,
    batch_size: int = 1024,
) -> None:
    """Reward-align independent trajectories without per-transition Python calls."""

    from src.tasks.rwm_velocity.mdp.rewards import (
        Go2RWMRewardState,
        compute_go2_imagination_reward,
    )

    if batch_size < 1:
        raise ValueError("Reward-alignment batch size must be positive.")
    groups: dict[int, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        groups.setdefault(len(trajectory["states"]), []).append(trajectory)

    processed = 0
    for trajectory_length, group in sorted(groups.items()):
        for offset in range(0, len(group), batch_size):
            chunk = group[offset : offset + batch_size]
            states = torch.stack([trajectory["states"] for trajectory in chunk])
            next_states = torch.stack([trajectory["next_states"] for trajectory in chunk])
            actions = torch.stack([trajectory["actions"] for trajectory in chunk])
            previous_actions = torch.stack([trajectory["prev_actions"] for trajectory in chunk])
            commands = torch.stack([trajectory["commands"] for trajectory in chunk])
            contacts = torch.stack([trajectory["contacts"] for trajectory in chunk])
            count = len(chunk)
            reward_state = Go2RWMRewardState.create(
                num_envs=count,
                action_dim=int(actions.shape[-1]),
                device="cpu",
                step_dt=float(step_dt),
                reward_version=reward_version,
            )
            reward_state.weights.command_response = float(reward_command_response_weight)
            reward_state.weights.yaw_command_response = float(reward_yaw_command_response_weight)
            reward_state.weights.wrong_direction = float(reward_wrong_direction_weight)
            reward_state.weights.response_shortfall = float(reward_response_shortfall_weight)
            reward_state.weights.response_floor = float(reward_response_floor)
            reward_state.weights.active_command_bias = float(reward_active_command_bias)
            reward_state.weights.uncertainty = float(reward_uncertainty_penalty_weight)
            reward_state.weights.action_rate_l2 = float(reward_action_rate_l2)
            reward_state.weights.dof_acc_l2 = float(reward_dof_acc_l2)
            reward_state.weights.dof_torques_l2 = float(reward_dof_torques_l2)
            reward_state.command_active_threshold = float(reward_command_active_threshold)
            reward_state.motion_gate_low = float(reward_motion_gate_low)
            reward_state.motion_gate_high = float(reward_motion_gate_high)
            reward_state.last_joint_vel.copy_(states[:, 0, 21:33])
            reward_state.last_action.copy_(previous_actions[:, 0])
            rewards_by_step: list[torch.Tensor] = []
            uncertainty = torch.zeros(count)
            for index in range(trajectory_length):
                reward, _ = compute_go2_imagination_reward(
                    state=next_states[:, index],
                    action=actions[:, index],
                    command=commands[:, index],
                    foot_contact=contacts[:, index],
                    episode_length=torch.full((count,), index, dtype=torch.long),
                    reward_state=reward_state,
                    epistemic_uncertainty=uncertainty,
                )
                rewards_by_step.append(reward)
            aligned = torch.stack(rewards_by_step, dim=1)
            for local_index, trajectory in enumerate(chunk):
                trajectory["dataset_rewards"] = trajectory["rewards"].clone()
                trajectory["rewards"] = _apply_replay_safety_costs(
                    aligned[local_index],
                    trajectory["actions"],
                    trajectory["prev_actions"],
                    trajectory["terminations"],
                    terminal_penalty=terminal_penalty,
                    override_terminal_reward=override_terminal_reward,
                    failure_backprop_steps=failure_backprop_steps,
                    failure_backprop_penalty=failure_backprop_penalty,
                    action_saturation_threshold=action_saturation_threshold,
                    action_saturation_penalty_scale=action_saturation_penalty_scale,
                    action_delta_penalty_scale=action_delta_penalty_scale,
                )
            processed += count
            print(
                f"[v10 replay] reward-aligned {processed}/{len(trajectories)} trajectories",
                flush=True,
            )


def _n_step_rows(trajectory: dict[str, Any], n_step: int, gamma: float) -> dict[str, list[torch.Tensor]]:
    states = trajectory["states"]
    actions = trajectory["actions"]
    next_states = trajectory["next_states"]
    commands = trajectory["commands"]
    previous_actions = trajectory["prev_actions"]
    rewards = trajectory["rewards"].reshape(-1).float()
    terminated = trajectory["terminations"].reshape(len(states), -1).bool().any(dim=-1)
    rows = {key: [] for key in ("observation", "action", "reward", "terminated", "truncated", "next_observation")}
    # A non-terminal row needs n rewards and bootstraps from next_state[final].
    # At a terminal boundary bootstrap is zero, so also flush the shorter
    # 2-step and 1-step suffixes.  This retains the action that directly caused
    # the failure instead of assigning every terminal only to action L-n.
    row_count = (
        len(states)
        if len(states) > 0 and bool(terminated[-1])
        else max(0, len(states) - n_step + 1)
    )
    for start in range(row_count):
        total_reward = torch.tensor(0.0)
        final = start
        final_terminated = False
        for offset in range(n_step):
            index = start + offset
            if index >= len(states):
                break
            total_reward += (gamma**offset) * rewards[index]
            final = index
            final_terminated = bool(terminated[index])
            if final_terminated:
                break
        rows["observation"].append(_make_policy_obs(states[start], commands[start], previous_actions[start]))
        rows["action"].append(actions[start].float())
        rows["reward"].append(total_reward.float())
        rows["terminated"].append(torch.tensor(final_terminated, dtype=torch.float32))
        rows["truncated"].append(torch.tensor(False, dtype=torch.float32))
        next_command = commands[final] if final_terminated or final + 1 >= len(commands) else commands[final + 1]
        rows["next_observation"].append(_make_policy_obs(next_states[final], next_command, actions[final]))
    return rows


def _add_action_safety_summary(
    summary: dict[str, Any],
    trajectory: dict[str, Any],
    saturation_threshold: float,
) -> None:
    actions = trajectory["actions"].float()
    saturated = (torch.abs(actions) >= float(saturation_threshold)).any(dim=-1)
    previous_actions = trajectory["prev_actions"].float()
    summary["action_saturation_fraction"] = float(saturated.float().mean())
    summary["action_abs_mean"] = float(torch.abs(actions).mean())
    summary["action_delta_abs_mean"] = float(torch.abs(actions - previous_actions).mean())


def main() -> None:
    args = parse_args()
    protocol, protocol_path, protocol_sha = load_protocol(args.protocol)
    metric_config = None
    metric_config_path = None
    metric_config_sha = None
    if args.metric_pilot_config:
        metric_config, metric_config_path, metric_config_sha = load_metric_config(
            args.metric_pilot_config,
            base_protocol_sha256=protocol_sha,
        )
        if args.selection not in {"metric", "random"}:
            raise ValueError("Metric-pilot config only supports metric or random selection.")
        if int(metric_config["selection"]["selected_trajectory_count"]) != int(
            protocol["candidate"]["selected_trajectory_count"]
        ):
            raise ValueError("Metric-pilot and V10 selected trajectory counts differ.")
        setattr(args, "_metric_pilot_config", metric_config)
    args.command_motion_gate = True
    validity = protocol["validity"]
    frozen_values = {
        "min_linear_realization_ratio": validity["minimum_linear_realization_ratio"],
        "min_yaw_realization_ratio": validity["minimum_yaw_realization_ratio"],
        "min_linear_net_displacement": validity["minimum_linear_net_displacement"],
        "min_yaw_net_change": validity["minimum_yaw_net_change"],
        "max_command_direction_violation_rate": validity["maximum_direction_violation_rate"],
        "max_hard_direction_violation_rate": validity["maximum_direction_violation_rate"],
        "action_saturation_threshold": validity["action_saturation_threshold"],
        "max_saturation_fraction": validity["maximum_saturation_fraction"],
        "max_reset_reconstruction_error": validity["maximum_reset_reconstruction_error"],
        "minimum_terminal_length": validity["minimum_terminal_length"],
        "trajectory_length": protocol["candidate"]["horizon"],
        "trajectory_stride": protocol["candidate"]["horizon"],
        "n_step": protocol["replay"]["n_step"],
        "gamma": protocol["replay"]["gamma"],
    }
    mismatches = {
        name: {"actual": getattr(args, name), "required": required}
        for name, required in frozen_values.items()
        if abs(float(getattr(args, name)) - float(required)) > 1.0e-12
    }
    if mismatches:
        raise ValueError(f"TRACE V10 CLI values conflict with the frozen protocol: {mismatches}")
    if args.selection_scope != "global":
        raise ValueError("TRACE V10 only supports global selection.")
    if abs(float(args.select_ratio) - float(protocol["candidate"]["selection_ratio"])) > 1.0e-12:
        raise ValueError("TRACE V10 select_ratio must match the frozen protocol.")
    if args.failure_transition_ratio not in (None, 0.0) or float(args.failure_trajectory_ratio) != 0.0:
        raise ValueError("TRACE V10 disables forced failure quotas; natural terminations remain eligible.")
    if (
        args.override_terminal_reward
        or args.failure_backprop_steps != 0
        or args.failure_backprop_penalty != 0.0
        or args.action_saturation_penalty_scale != 0.0
        or args.action_delta_penalty_scale != 0.0
    ):
        raise ValueError("TRACE V10 replay reward must match the frozen protocol without extra shaping.")
    if not 0.0 < args.select_ratio <= 1.0:
        raise ValueError("--select_ratio must be in (0, 1].")
    if not 0.0 <= args.failure_trajectory_ratio <= 1.0:
        raise ValueError("--failure_trajectory_ratio must be in [0, 1].")
    if args.failure_transition_ratio is not None and not 0.0 <= args.failure_transition_ratio <= 1.0:
        raise ValueError("--failure_transition_ratio must be in [0, 1].")
    if args.failure_backprop_steps < 0 or args.failure_backprop_penalty < 0.0:
        raise ValueError("Failure backprop steps and penalty must be non-negative.")
    if not 0.0 < args.action_saturation_threshold <= 1.0:
        raise ValueError("--action_saturation_threshold must be in (0, 1].")
    if not 0.0 <= args.max_saturation_fraction <= 1.0:
        raise ValueError("--max_saturation_fraction must be in [0, 1].")
    if args.max_reset_reconstruction_error < 0.0:
        raise ValueError("--max_reset_reconstruction_error must be non-negative.")
    if args.action_saturation_penalty_scale < 0.0 or args.action_delta_penalty_scale < 0.0:
        raise ValueError("Action penalty scales must be non-negative.")
    if args.trajectory_length < 1 or args.trajectory_stride < 1 or args.n_step < 1:
        raise ValueError("Trajectory length, stride, and n_step must be positive.")
    for name in (
        "min_linear_realization_ratio",
        "min_yaw_realization_ratio",
        "max_command_direction_violation_rate",
        "max_hard_direction_violation_rate",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}.")
    if not args.summaries_only and args.trajectory_length <= args.n_step:
        raise ValueError("trajectory_length must be greater than n_step when building replay.")
    print(f"[v10 replay] loading candidates: {args.candidate_dataset}", flush=True)
    dataset = torch.load(args.candidate_dataset, map_location="cpu", weights_only=False)
    _validate_controlled_candidates(dataset, allow_legacy=bool(args.allow_legacy_candidates))
    trace_candidate_metadata = dict((dataset.get("metadata") or {}).get("trace_candidates") or {})
    expected_candidate_metadata = {
        "action_temperature": float(protocol["candidate"]["action_temperature"]),
        "rollout_length": int(protocol["candidate"]["horizon"]),
        "trajectories_per_state": int(protocol["candidate"]["trajectories_per_start"]),
    }
    candidate_metadata_mismatches = {
        key: {"actual": trace_candidate_metadata.get(key), "required": required}
        for key, required in expected_candidate_metadata.items()
        if trace_candidate_metadata.get(key) != required
    }
    if candidate_metadata_mismatches:
        raise ValueError(
            "TRACE candidate collection differs from the frozen V10 protocol: "
            f"{candidate_metadata_mismatches}"
        )
    print("[v10 replay] extracting candidate trajectories", flush=True)
    candidates = _candidate_windows(
        dataset,
        args.trajectory_length,
        args.trajectory_stride,
        include_terminal_prefixes=bool(args.include_terminal_prefixes),
        minimum_terminal_length=int(args.minimum_terminal_length),
    )
    for candidate in candidates:
        candidate["nominal_horizon"] = int(protocol["candidate"]["horizon"])
    expected_candidates = int(protocol["candidate"]["candidate_trajectory_count"])
    if metric_config is None and len(candidates) != expected_candidates:
        raise ValueError(
            f"TRACE V10 requires exactly {expected_candidates} candidate trajectories, got {len(candidates)}."
        )
    if metric_config is not None:
        selected_target = int(metric_config["selection"]["selected_trajectory_count"])
        if not selected_target <= len(candidates) <= expected_candidates:
            raise ValueError(
                "Metric pilot must retain enough complete/terminal-prefix trajectories for its fixed "
                f"global selection: candidates={len(candidates)}, selected_target={selected_target}, "
                f"rollout_attempts={expected_candidates}."
            )
    print(f"[v10 replay] extracted {len(candidates)} trajectories", flush=True)
    if args.reward_source == "rwm_aligned":
        step_dt = float((dataset.get("metadata") or {}).get("dt", 0.02))
        _align_rewards_with_rwm_batch(
            candidates,
            step_dt,
            float(args.terminal_penalty),
            override_terminal_reward=bool(args.override_terminal_reward),
            failure_backprop_steps=int(args.failure_backprop_steps),
            failure_backprop_penalty=float(args.failure_backprop_penalty),
            action_saturation_threshold=float(args.action_saturation_threshold),
            action_saturation_penalty_scale=float(args.action_saturation_penalty_scale),
            action_delta_penalty_scale=float(args.action_delta_penalty_scale),
            reward_version=str(args.reward_version),
            reward_command_response_weight=float(args.reward_command_response_weight),
            reward_yaw_command_response_weight=float(args.reward_yaw_command_response_weight),
            reward_wrong_direction_weight=float(args.reward_wrong_direction_weight),
            reward_response_shortfall_weight=float(args.reward_response_shortfall_weight),
            reward_response_floor=float(args.reward_response_floor),
            reward_active_command_bias=float(args.reward_active_command_bias),
            reward_uncertainty_penalty_weight=float(args.reward_uncertainty_penalty_weight),
            reward_action_rate_l2=float(args.reward_action_rate_l2),
            reward_dof_acc_l2=float(args.reward_dof_acc_l2),
            reward_dof_torques_l2=float(args.reward_dof_torques_l2),
            reward_command_active_threshold=float(args.reward_command_active_threshold),
            reward_motion_gate_low=float(args.reward_motion_gate_low),
            reward_motion_gate_high=float(args.reward_motion_gate_high),
        )
    print("[v10 replay] computing trajectory summaries", flush=True)
    summaries = [summarize_go2_trajectory(candidate) for candidate in candidates]
    trace_metadata = dict((dataset.get("metadata") or {}).get("trace_candidates") or {})
    reset_dataset = trace_metadata.get("reset_dataset")
    reset_path = Path(reset_dataset).expanduser() if reset_dataset else None
    source_namespace = (
        sha256_path(reset_path)[:16]
        if reset_path is not None and reset_path.is_file()
        else str(trace_metadata.get("reset_dataset_sha256", ""))[:16]
    )
    if not source_namespace:
        raise ValueError(
            "TRACE candidate metadata cannot identify the reset dataset. A stable source namespace "
            "is required to prevent scorer train/validation leakage across refreshes."
        )
    candidate_namespace = sha256_path(args.candidate_dataset)[:16]
    for summary, candidate in zip(summaries, candidates, strict=True):
        summary["start_state_key"] = f"{candidate_namespace}:{int(candidate['start_state_id'])}"
        summary["source_state_key"] = f"{source_namespace}:{int(candidate['start_state_id'])}"
        summary["comparison_group_key"] = summary["source_state_key"]
        summary["candidate_namespace"] = candidate_namespace
        _add_action_safety_summary(summary, candidate, args.action_saturation_threshold)
    if args.reference_summaries:
        summaries = attach_references_to_summaries(
            summaries,
            load_reference_map(args.reference_summaries),
            strict=True,
        )
    for summary in summaries:
        rejection = _motion_gate_rejection_reason(summary, args)
        summary["command_motion_gate_passed"] = rejection is None
        summary["command_motion_gate_rejection_reason"] = rejection
        summary["command_motion_gate_tier"] = (
            "strict"
            if rejection is None
            else "rejected"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_jsonl)
        if args.summary_jsonl
        else output.with_suffix(".summaries.jsonl")
    )
    if args.summaries_only:
        with summary_path.open("w", encoding="utf-8") as handle:
            for summary in summaries:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")
        print(f"Wrote {len(summaries)} candidate summaries to {summary_path}")
        return
    print(f"[v10 replay] selecting trajectories with {args.selection}", flush=True)
    selected_indices, scores = _select_indices(summaries, args)
    selection_diagnostics = dict(getattr(args, "_selection_diagnostics", {}))
    expected_selected = selection_diagnostics.get("target_selected_count")
    if expected_selected is not None and len(selected_indices) != int(expected_selected):
        raise RuntimeError(
            f"Replay selection silently changed size: selected={len(selected_indices)}, target={expected_selected}."
        )
    selected_set = set(int(index) for index in selected_indices)
    for index, summary in enumerate(summaries):
        summary["trace_score"] = float(scores[index])
        summary["selected"] = index in selected_set

    selected_signed_bucket_counts = np.zeros((len(MODE_TO_ID), 3, 6), dtype=np.int64)
    for index in selected_indices:
        summary = summaries[int(index)]
        mode_id = MODE_TO_ID[str(summary["command_mode"])]
        buckets = signed_magnitude_buckets(
            (
                float(summary["command_vx_mean"]),
                float(summary["command_vy_mean"]),
                float(summary["command_yaw_mean"]),
            ),
            protocol,
        )
        for axis, bucket in enumerate(buckets):
            if bucket >= 0:
                selected_signed_bucket_counts[mode_id, axis, bucket] += 1

    replay_rows = {
        key: []
        for key in (
            "observation",
            "action",
            "reward",
            "terminated",
            "truncated",
            "next_observation",
        )
    }
    replay_aux_rows: dict[str, list[torch.Tensor]] = {
        "command_mode_id": [],
        "command_signed_buckets": [],
        "source_state_id": [],
        "refresh_cycle": [],
    }
    failure_source_transition_count = 0
    for index in selected_indices:
        rows = _n_step_rows(candidates[int(index)], args.n_step, args.gamma)
        if bool(summaries[int(index)].get("terminal_flag", False)):
            failure_source_transition_count += len(rows["reward"])
        for key, values in rows.items():
            replay_rows[key].extend(values)
        row_count = len(rows["reward"])
        summary = summaries[int(index)]
        command = (
            float(summary["command_vx_mean"]),
            float(summary["command_vy_mean"]),
            float(summary["command_yaw_mean"]),
        )
        mode_id = MODE_TO_ID[str(summary["command_mode"])]
        buckets = signed_magnitude_buckets(command, protocol)
        replay_aux_rows["command_mode_id"].extend(
            torch.tensor(mode_id, dtype=torch.int16) for _ in range(row_count)
        )
        replay_aux_rows["command_signed_buckets"].extend(
            torch.tensor(buckets, dtype=torch.int8) for _ in range(row_count)
        )
        replay_aux_rows["source_state_id"].extend(
            torch.tensor(int(summary["start_state_id"]), dtype=torch.int64) for _ in range(row_count)
        )
        replay_aux_rows["refresh_cycle"].extend(
            torch.tensor(int(args.refresh_cycle), dtype=torch.int16) for _ in range(row_count)
        )
    replay = {key: torch.stack(values) for key, values in replay_rows.items()}
    replay.update({key: torch.stack(values) for key, values in replay_aux_rows.items()})
    artifact = {
        "format_version": "go2_trace_replay_v10_shard_v1",
        **replay,
        "metadata": {
            "trace_protocol_version": "go2_trace_v10",
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_sha,
            "refresh_cycle": int(args.refresh_cycle),
            "target_dataset": str(Path(args.target_dataset).resolve()),
            "target_dataset_sha256": sha256_path(args.target_dataset),
            "target_mode_bucket_probabilities": dataset_mode_bucket_probabilities(
                args.target_dataset, protocol
            ),
            "policy_checkpoint": str(Path(args.policy_checkpoint).resolve()),
            "policy_checkpoint_sha256": sha256_path(args.policy_checkpoint),
            "candidate_dataset": str(Path(args.candidate_dataset).resolve()),
            "candidate_dataset_sha256": sha256_path(args.candidate_dataset),
            "reference_summaries": (
                str(Path(args.reference_summaries).resolve()) if args.reference_summaries else None
            ),
            "reference_summaries_sha256": (
                sha256_path(args.reference_summaries) if args.reference_summaries else None
            ),
            "scorer_checkpoint": str(Path(args.scorer_checkpoint).resolve()) if args.scorer_checkpoint else None,
            "scorer_checkpoint_sha256": sha256_path(args.scorer_checkpoint) if args.scorer_checkpoint else None,
            "metric_pilot_config": str(metric_config_path) if metric_config_path else None,
            "metric_pilot_config_sha256": metric_config_sha,
            "sampling_strategy": (
                str(metric_config["selection"]["sampling_strategy"])
                if metric_config is not None
                else "dataset_mode_signed_magnitude_stratified"
            ),
            "selection": args.selection,
            "selection_scope": str(args.selection_scope),
            "select_ratio": float(args.select_ratio),
            "per_start_target_count": int(args.per_start_target_count),
            "command_motion_gate": bool(args.command_motion_gate),
            "min_linear_realization_ratio": float(args.min_linear_realization_ratio),
            "min_yaw_realization_ratio": float(args.min_yaw_realization_ratio),
            "max_command_direction_violation_rate": float(args.max_command_direction_violation_rate),
            "max_hard_direction_violation_rate": float(args.max_hard_direction_violation_rate),
            "command_mode_weights": _parse_command_mode_weights(args.command_mode_weights),
            "trajectory_length": int(args.trajectory_length),
            "trajectory_stride": int(args.trajectory_stride),
            "include_terminal_prefixes": bool(args.include_terminal_prefixes),
            "minimum_terminal_length": int(args.minimum_terminal_length),
            "failure_trajectory_ratio": float(args.failure_trajectory_ratio),
            "failure_transition_ratio_target": (
                None if args.failure_transition_ratio is None else float(args.failure_transition_ratio)
            ),
            "failure_quota_mode": (
                "disabled_natural"
                if (
                    (args.failure_transition_ratio is None and float(args.failure_trajectory_ratio) <= 0.0)
                    or (
                        args.failure_transition_ratio is not None
                        and float(args.failure_transition_ratio) <= 0.0
                    )
                )
                else "forced_target"
            ),
            "terminal_penalty": float(args.terminal_penalty),
            "override_terminal_reward": bool(args.override_terminal_reward),
            "failure_backprop_steps": int(args.failure_backprop_steps),
            "failure_backprop_penalty": float(args.failure_backprop_penalty),
            "action_saturation_threshold": float(args.action_saturation_threshold),
            "max_saturation_fraction": float(args.max_saturation_fraction),
            "action_saturation_penalty_scale": float(args.action_saturation_penalty_scale),
            "action_delta_penalty_scale": float(args.action_delta_penalty_scale),
            "n_step": int(args.n_step),
            "gamma": float(args.gamma),
            "reward_source": str(args.reward_source),
            "reward_version": str(args.reward_version),
            "reward_command_response_weight": float(args.reward_command_response_weight),
            "reward_yaw_command_response_weight": float(args.reward_yaw_command_response_weight),
            "reward_wrong_direction_weight": float(args.reward_wrong_direction_weight),
            "reward_response_shortfall_weight": float(args.reward_response_shortfall_weight),
            "reward_response_floor": float(args.reward_response_floor),
            "reward_active_command_bias": float(args.reward_active_command_bias),
            "reward_uncertainty_penalty_weight": float(args.reward_uncertainty_penalty_weight),
            "reward_action_rate_l2": float(args.reward_action_rate_l2),
            "reward_dof_acc_l2": float(args.reward_dof_acc_l2),
            "reward_dof_torques_l2": float(args.reward_dof_torques_l2),
            "reward_command_active_threshold": float(args.reward_command_active_threshold),
            "reward_motion_gate_low": float(args.reward_motion_gate_low),
            "reward_motion_gate_high": float(args.reward_motion_gate_high),
            "reward_alignment_scope": "common_go2_reward_version",
            "epistemic_uncertainty_source": "not_applicable_true_simulator_transition",
            "epistemic_uncertainty_value": 0.0,
            "rwm_buffer_may_apply_model_uncertainty_penalty": True,
            "candidate_count": len(candidates),
            "candidate_rollout_attempt_count": expected_candidates,
            "safety_eligible_candidate_count": int(
                sum(
                    not bool(summary.get("nonfinite_flag", True))
                    and np.isfinite(float(summary.get("reset_reconstruction_error", float("nan"))))
                    and float(summary["reset_reconstruction_error"])
                    <= float(args.max_reset_reconstruction_error)
                    and float(summary["action_saturation_fraction"])
                    <= float(args.max_saturation_fraction)
                    for summary in summaries
                )
            ),
            "selected_trajectory_count": len(selected_indices),
            "selection_diagnostics": selection_diagnostics,
            "command_motion_gate_eligible_count": int(
                sum(bool(summary["command_motion_gate_passed"]) for summary in summaries)
            ),
            "command_motion_gate_rejection_counts": {
                reason: int(
                    sum(summary["command_motion_gate_rejection_reason"] == reason for summary in summaries)
                )
                for reason in (
                    "severe_direction_violation",
                    "direction_violation",
                    "linear_realization",
                    "yaw_realization",
                )
            },
            "candidate_command_mode_counts": {
                mode: int(sum(summary.get("command_mode") == mode for summary in summaries))
                for mode in sorted({str(summary.get("command_mode", "unknown")) for summary in summaries})
            },
            "selected_command_mode_counts": {
                mode: int(sum(summaries[int(index)].get("command_mode") == mode for index in selected_indices))
                for mode in sorted({str(summary.get("command_mode", "unknown")) for summary in summaries})
            },
            "selected_signed_bucket_counts": selected_signed_bucket_counts.tolist(),
            "candidate_terminal_trajectory_count": int(
                sum(bool(summary.get("terminal_flag", False)) for summary in summaries)
            ),
            "selected_terminal_trajectory_count": int(
                sum(bool(summaries[int(index)].get("terminal_flag", False)) for index in selected_indices)
            ),
            "transition_count": int(replay["reward"].shape[0]),
            "failure_source_transition_count": int(failure_source_transition_count),
            "failure_source_transition_ratio": float(
                failure_source_transition_count / max(int(replay["reward"].shape[0]), 1)
            ),
            "observation_dim": int(replay["observation"].shape[-1]),
            "action_dim": int(replay["action"].shape[-1]),
            "candidate_reset_mode": dict((dataset.get("metadata") or {}).get("trace_candidates") or {}).get(
                "reset_mode"
            ),
        },
    }
    print(
        f"[v10 replay] saving {len(selected_indices)} trajectories / "
        f"{int(replay['reward'].shape[0])} transitions to {output}",
        flush=True,
    )
    torch.save(artifact, output)
    with summary_path.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
