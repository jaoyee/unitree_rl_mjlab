#!/usr/bin/env python3
"""Build safety-filtered TRACE replay with dense pre-terminal failure costs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.scorer import load_scorer_checkpoint, score_summaries
from scripts.reinforcement_learning.rwm_trace.trajectory import summarize_go2_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate_dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer_checkpoint", default=None)
    parser.add_argument("--selection", choices=("scorer", "random", "all"), default="scorer")
    parser.add_argument("--select_ratio", type=float, default=0.1)
    parser.add_argument("--trajectory_length", type=int, default=20)
    parser.add_argument("--trajectory_stride", type=int, default=20)
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
        help="Reward assigned to an environment-terminal transition after reward alignment.",
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
        default=1.0,
        help="Reject trajectories exceeding this fraction of saturated transitions.",
    )
    parser.add_argument("--action_saturation_penalty_scale", type=float, default=0.0)
    parser.add_argument("--action_delta_penalty_scale", type=float, default=0.0)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_source", choices=("rwm_aligned", "dataset"), default="rwm_aligned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_jsonl", default=None)
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
        if not source_identity.get("available", False) or not source_identity.get("passed", False):
            errors.append("exact snapshot did not reproduce the dataset source next_state")
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
    eligible_indices = np.asarray(
        [
            index
            for index, summary in enumerate(summaries)
            if float(summary.get("action_saturation_fraction", 0.0))
            <= float(args.max_saturation_fraction)
        ],
        dtype=np.int64,
    )
    if len(eligible_indices) == 0:
        raise ValueError("Safety filter rejected every candidate trajectory.")
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
    selected_count = max(1, int(math.ceil(len(eligible_indices) * float(args.select_ratio))))
    jitter = rng.uniform(0.0, 1.0e-9, size=count)
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
    ranked = eligible_indices[np.argsort(-(scores[eligible_indices] + jitter[eligible_indices]))]
    if quota_disabled:
        # A zero target disables forced failure composition. Natural terminal
        # trajectories remain eligible and are selected solely by the chosen
        # scorer/random ranking.
        return ranked[:selected_count].astype(np.int64), scores
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


def _apply_replay_safety_costs(
    rewards: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    terminated: torch.Tensor,
    *,
    terminal_penalty: float,
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
    rewards[terminated] = float(terminal_penalty)
    return rewards


def _align_rewards_with_rwm(
    trajectory: dict[str, Any],
    step_dt: float,
    terminal_penalty: float,
    *,
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
        failure_backprop_steps=failure_backprop_steps,
        failure_backprop_penalty=failure_backprop_penalty,
        action_saturation_threshold=action_saturation_threshold,
        action_saturation_penalty_scale=action_saturation_penalty_scale,
        action_delta_penalty_scale=action_delta_penalty_scale,
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
    saturated = (torch.abs(actions) > float(saturation_threshold)).any(dim=-1)
    previous_actions = trajectory["prev_actions"].float()
    summary["action_saturation_fraction"] = float(saturated.float().mean())
    summary["action_abs_mean"] = float(torch.abs(actions).mean())
    summary["action_delta_abs_mean"] = float(torch.abs(actions - previous_actions).mean())


def main() -> None:
    args = parse_args()
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
    if args.action_saturation_penalty_scale < 0.0 or args.action_delta_penalty_scale < 0.0:
        raise ValueError("Action penalty scales must be non-negative.")
    if args.trajectory_length < 1 or args.trajectory_stride < 1 or args.n_step < 1:
        raise ValueError("Trajectory length, stride, and n_step must be positive.")
    if not args.summaries_only and args.trajectory_length <= args.n_step:
        raise ValueError("trajectory_length must be greater than n_step when building replay.")
    dataset = torch.load(args.candidate_dataset, map_location="cpu", weights_only=False)
    _validate_controlled_candidates(dataset, allow_legacy=bool(args.allow_legacy_candidates))
    candidates = _candidate_windows(
        dataset,
        args.trajectory_length,
        args.trajectory_stride,
        include_terminal_prefixes=bool(args.include_terminal_prefixes),
        minimum_terminal_length=int(args.minimum_terminal_length),
    )
    if args.reward_source == "rwm_aligned":
        step_dt = float((dataset.get("metadata") or {}).get("dt", 0.02))
        for candidate in candidates:
            _align_rewards_with_rwm(
                candidate,
                step_dt,
                float(args.terminal_penalty),
                failure_backprop_steps=int(args.failure_backprop_steps),
                failure_backprop_penalty=float(args.failure_backprop_penalty),
                action_saturation_threshold=float(args.action_saturation_threshold),
                action_saturation_penalty_scale=float(args.action_saturation_penalty_scale),
                action_delta_penalty_scale=float(args.action_delta_penalty_scale),
            )
    summaries = [summarize_go2_trajectory(candidate) for candidate in candidates]
    candidate_namespace = sha256_path(args.candidate_dataset)[:16]
    for summary, candidate in zip(summaries, candidates, strict=True):
        summary["start_state_key"] = f"{candidate_namespace}:{int(candidate['start_state_id'])}"
        summary["comparison_group_key"] = summary["start_state_key"]
        summary["candidate_namespace"] = candidate_namespace
        _add_action_safety_summary(summary, candidate, args.action_saturation_threshold)
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
    selected_indices, scores = _select_indices(summaries, args)
    selected_set = set(int(index) for index in selected_indices)
    for index, summary in enumerate(summaries):
        summary["trace_score"] = float(scores[index])
        summary["selected"] = index in selected_set

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
    failure_source_transition_count = 0
    for index in selected_indices:
        rows = _n_step_rows(candidates[int(index)], args.n_step, args.gamma)
        if bool(summaries[int(index)].get("terminal_flag", False)):
            failure_source_transition_count += len(rows["reward"])
        for key, values in rows.items():
            replay_rows[key].extend(values)
    replay = {key: torch.stack(values) for key, values in replay_rows.items()}
    artifact = {
        "format_version": "go2_trace_replay_v2",
        **replay,
        "metadata": {
            "trace_protocol_version": "go2_trace_v5_controlled",
            "candidate_dataset": str(Path(args.candidate_dataset).resolve()),
            "candidate_dataset_sha256": sha256_path(args.candidate_dataset),
            "scorer_checkpoint": str(Path(args.scorer_checkpoint).resolve()) if args.scorer_checkpoint else None,
            "scorer_checkpoint_sha256": sha256_path(args.scorer_checkpoint) if args.scorer_checkpoint else None,
            "selection": args.selection,
            "select_ratio": float(args.select_ratio),
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
            "failure_backprop_steps": int(args.failure_backprop_steps),
            "failure_backprop_penalty": float(args.failure_backprop_penalty),
            "action_saturation_threshold": float(args.action_saturation_threshold),
            "max_saturation_fraction": float(args.max_saturation_fraction),
            "action_saturation_penalty_scale": float(args.action_saturation_penalty_scale),
            "action_delta_penalty_scale": float(args.action_delta_penalty_scale),
            "n_step": int(args.n_step),
            "gamma": float(args.gamma),
            "reward_source": str(args.reward_source),
            "candidate_count": len(candidates),
            "safety_eligible_candidate_count": int(
                sum(
                    float(summary["action_saturation_fraction"])
                    <= float(args.max_saturation_fraction)
                    for summary in summaries
                )
            ),
            "selected_trajectory_count": len(selected_indices),
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
    torch.save(artifact, output)
    with summary_path.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
