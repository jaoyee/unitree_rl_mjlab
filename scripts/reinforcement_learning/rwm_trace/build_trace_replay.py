#!/usr/bin/env python3
"""Build a score-selected replay artifact from Go2 simulator candidates."""

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
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_source", choices=("rwm_aligned", "dataset"), default="rwm_aligned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_jsonl", default=None)
    parser.add_argument("--summaries_only", action="store_true")
    return parser.parse_args()


def _make_policy_obs(state: torch.Tensor, command: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    return torch.cat([state[..., :33], command, previous_action], dim=-1).float()


def _candidate_windows(dataset: dict[str, Any], length: int, stride: int) -> list[dict[str, Any]]:
    tensors = {
        key: stack_time_key(dataset, key)
        for key in ("states", "actions", "next_states", "contacts", "terminations", "commands", "rewards")
    }
    previous_actions = (
        stack_time_key(dataset, "prev_actions")
        if dataset.get("prev_actions") is not None
        else torch.cat([torch.zeros_like(tensors["actions"][:1]), tensors["actions"][:-1]], dim=0)
    )
    episode_ids = stack_time_key(dataset, "episode_ids") if dataset.get("episode_ids") is not None else None
    trace_start_state_ids = dataset.get("trace_start_state_ids")
    if trace_start_state_ids is not None:
        trace_start_state_ids = torch.as_tensor(trace_start_state_ids).reshape(-1)
    reset_errors = (
        stack_time_key(dataset, "trace_reset_reconstruction_errors")
        if dataset.get("trace_reset_reconstruction_errors") is not None
        else None
    )
    mismatch = {
        "target_gap": dict((dataset.get("metadata") or {}).get("target_gap") or {}),
        "env_randomization": dict((dataset.get("metadata") or {}).get("env_randomization") or {}),
    }
    time_steps, num_envs = tensors["states"].shape[:2]
    candidates: list[dict[str, Any]] = []
    for env_id in range(num_envs):
        for start in range(0, time_steps - length + 1, stride):
            stop = start + length
            term = tensors["terminations"][start:stop, env_id].bool().reshape(length, -1).any(dim=-1)
            if term[:-1].any():
                continue
            if episode_ids is not None:
                ids = episode_ids[start:stop, env_id].reshape(length, -1)[:, 0]
                if (ids[1:] != ids[:-1]).any():
                    continue
            trajectory = {key: value[start:stop, env_id] for key, value in tensors.items()}
            trajectory["prev_actions"] = previous_actions[start:stop, env_id]
            trajectory["trajectory_id"] = f"env{env_id:05d}_start{start:08d}"
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
    return candidates


def _select_indices(
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(summaries)
    if count == 0:
        raise ValueError("Candidate dataset contains no complete trajectory windows.")
    if args.selection == "all":
        scores = np.zeros(count, dtype=np.float32)
        return np.arange(count), scores
    rng = np.random.default_rng(args.seed)
    if args.selection == "random":
        scores = rng.random(count, dtype=np.float32)
    else:
        if not args.scorer_checkpoint:
            raise ValueError("--scorer_checkpoint is required for scorer selection.")
        model, stats, _ = load_scorer_checkpoint(args.scorer_checkpoint)
        scores = score_summaries(model, summaries, stats)
    selected_count = max(1, int(math.ceil(count * float(args.select_ratio))))
    jitter = rng.uniform(0.0, 1.0e-9, size=count)
    selected = np.argsort(-(scores + jitter))[:selected_count]
    return selected.astype(np.int64), scores


def _align_rewards_with_rwm(trajectory: dict[str, Any], step_dt: float) -> None:
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


def _n_step_rows(trajectory: dict[str, Any], n_step: int, gamma: float) -> dict[str, list[torch.Tensor]]:
    states = trajectory["states"]
    actions = trajectory["actions"]
    next_states = trajectory["next_states"]
    commands = trajectory["commands"]
    previous_actions = trajectory["prev_actions"]
    rewards = trajectory["rewards"].reshape(-1).float()
    terminated = trajectory["terminations"].reshape(len(states), -1).bool().any(dim=-1)
    rows = {key: [] for key in ("observation", "action", "reward", "terminated", "truncated", "next_observation")}
    # A window boundary is not an environment terminal. Keep one extra row so
    # the bootstrap observation uses the command active after the final step.
    for start in range(max(0, len(states) - n_step)):
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
        rows["next_observation"].append(
            _make_policy_obs(next_states[final], commands[final + 1], actions[final])
        )
    return rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.select_ratio <= 1.0:
        raise ValueError("--select_ratio must be in (0, 1].")
    if args.trajectory_length < 1 or args.trajectory_stride < 1 or args.n_step < 1:
        raise ValueError("Trajectory length, stride, and n_step must be positive.")
    if not args.summaries_only and args.trajectory_length <= args.n_step:
        raise ValueError("trajectory_length must be greater than n_step when building replay.")
    dataset = torch.load(args.candidate_dataset, map_location="cpu", weights_only=False)
    candidates = _candidate_windows(dataset, args.trajectory_length, args.trajectory_stride)
    if args.reward_source == "rwm_aligned":
        step_dt = float((dataset.get("metadata") or {}).get("dt", 0.02))
        for candidate in candidates:
            _align_rewards_with_rwm(candidate, step_dt)
    summaries = [summarize_go2_trajectory(candidate) for candidate in candidates]
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
    for index in selected_indices:
        rows = _n_step_rows(candidates[int(index)], args.n_step, args.gamma)
        for key, values in rows.items():
            replay_rows[key].extend(values)
    replay = {key: torch.stack(values) for key, values in replay_rows.items()}
    artifact = {
        "format_version": "go2_trace_replay_v1",
        **replay,
        "metadata": {
            "candidate_dataset": str(Path(args.candidate_dataset).resolve()),
            "scorer_checkpoint": str(Path(args.scorer_checkpoint).resolve()) if args.scorer_checkpoint else None,
            "selection": args.selection,
            "select_ratio": float(args.select_ratio),
            "trajectory_length": int(args.trajectory_length),
            "trajectory_stride": int(args.trajectory_stride),
            "n_step": int(args.n_step),
            "gamma": float(args.gamma),
            "reward_source": str(args.reward_source),
            "candidate_count": len(candidates),
            "selected_trajectory_count": len(selected_indices),
            "transition_count": int(replay["reward"].shape[0]),
            "observation_dim": int(replay["observation"].shape[-1]),
            "action_dim": int(replay["action"].shape[-1]),
        },
    }
    torch.save(artifact, output)
    with summary_path.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
