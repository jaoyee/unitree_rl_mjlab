#!/usr/bin/env python3
"""Build same-start frozen-RWM reference summaries for Go2 TRACE candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.agent_proprioceptive import (
    create_go2_flashsac_proprioceptive_agent,
)
from scripts.reinforcement_learning.rwm_flashsac.dynamics_loader import (
    load_any_go2_dynamics_checkpoint,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    load_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    select_device,
    set_seed,
)
from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.build_trace_replay_v4 import (
    _add_action_safety_summary,
    _apply_replay_safety_costs,
)
from scripts.reinforcement_learning.rwm_trace.trajectory import summarize_go2_trajectory
from scripts.reinforcement_learning.rwm_trace.v10_protocol import load_protocol
from src.tasks.rwm_velocity.mdp.extractors import make_go2_policy_obs
from src.tasks.rwm_velocity.mdp.rewards import (
    Go2RWMRewardState,
    bad_orientation_from_state,
    compute_go2_imagination_reward,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate_dataset", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source_dataset", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_saturation_threshold", type=float, default=0.95)
    parser.add_argument("--terminal_penalty", type=float, default=-10.0)
    parser.add_argument(
        "--override_terminal_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--failure_backprop_steps", type=int, default=0)
    parser.add_argument("--failure_backprop_penalty", type=float, default=0.0)
    parser.add_argument("--action_saturation_penalty_scale", type=float, default=0.0)
    parser.add_argument("--action_delta_penalty_scale", type=float, default=0.0)
    parser.add_argument(
        "--allow_padded_history",
        action="store_true",
        help="Pad short source histories instead of failing. Diagnostic only; formal residual TRACE should avoid it.",
    )
    return parser.parse_args()


def _resolve_source_dataset(candidate: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return resolve_repo_path(explicit)
    metadata = dict(candidate.get("metadata") or {})
    trace = dict(metadata.get("trace_candidates") or {})
    value = trace.get("reset_dataset")
    if not value:
        raise ValueError("--source_dataset is required when candidate metadata lacks trace reset_dataset.")
    return resolve_repo_path(str(value))


def _load_policy(policy_path: Path, *, num_envs: int, action_dim: int, device: str) -> Any:
    cfg_path = policy_path / "rwm_flashsac_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected RWM policy config at {cfg_path}")
    cfg = load_config(cfg_path)
    cfg.agent.device_type = device
    cfg.agent.buffer_device_type = "cpu"
    cfg.agent.buffer_max_length = 1024
    cfg.agent.buffer_min_length = 1
    cfg.agent.sample_batch_size = 32
    cfg.agent.load_optimizer = False
    cfg.agent.load_reward_normalizer = False
    if device.startswith("cpu"):
        cfg.agent.use_amp = False
    obs_space, action_space = make_vector_spaces(num_envs, obs_dim=48, action_dim=action_dim)
    create_agent = (
        create_go2_flashsac_proprioceptive_agent
        if bool(OmegaConf.select(cfg, "agent.asymmetric_observation", default=False))
        else create_go2_flashsac_agent
    )
    agent = create_agent(obs_space, action_space, make_flashsac_config(cfg, device=device))
    agent.load(str(policy_path))
    return agent


def _source_tensors(source: dict[str, Any]) -> dict[str, torch.Tensor]:
    out = {
        "states": stack_time_key(source, "states").float(),
        "actions": stack_time_key(source, "actions").float(),
        "prev_actions": stack_time_key(source, "prev_actions").float(),
        "commands": stack_time_key(source, "commands").float(),
        "episode_ids": stack_time_key(source, "episode_ids").long(),
        "timesteps": stack_time_key(source, "timesteps").long(),
    }
    if out["states"].shape[-1] != 45:
        raise ValueError(f"Expected 45D source states, got {out['states'].shape[-1]}")
    return out


def _history_for_flat_id(
    tensors: dict[str, torch.Tensor],
    flat_id: int,
    history_horizon: int,
    *,
    allow_padded_history: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int]:
    states = tensors["states"]
    prev_actions = tensors["prev_actions"]
    commands = tensors["commands"]
    episode_ids = tensors["episode_ids"]
    timesteps = tensors["timesteps"]
    time_steps, num_envs = states.shape[:2]
    if flat_id < 0 or flat_id >= time_steps * num_envs:
        raise IndexError(f"Source flat_id={flat_id} outside [0, {time_steps * num_envs})")
    t = int(flat_id // num_envs)
    env_id = int(flat_id % num_envs)

    start = t
    while start > 0 and t - start + 1 < history_horizon:
        same_episode = bool(episode_ids[start - 1, env_id] == episode_ids[start, env_id])
        contiguous = bool(timesteps[start, env_id] == timesteps[start - 1, env_id] + 1)
        if not same_episode or not contiguous:
            break
        start -= 1
    valid_len = t - start + 1
    if valid_len < history_horizon and not allow_padded_history:
        raise ValueError(
            f"Source flat_id={flat_id} has only {valid_len} contiguous history rows; "
            f"need {history_horizon}. Increase TRACE_MIN_SOURCE_TIMESTEP or pass --allow_padded_history."
        )

    state_hist = states[start : t + 1, env_id]
    # Each state row is paired with the action that entered that state.  The
    # rollout loop shifts this history and appends the newly sampled action,
    # yielding actions[start:t+1] for the first RWM prediction.  Mixing source
    # actions with only the final prev_action duplicates action[t-1] and drops
    # the oldest action, silently misaligning the recurrent history.
    action_hist = prev_actions[start : t + 1, env_id].clone()
    padded = valid_len < history_horizon
    if padded:
        pad_count = history_horizon - valid_len
        state_hist = torch.cat([state_hist[:1].repeat(pad_count, 1), state_hist], dim=0)
        action_hist = torch.cat([action_hist[:1].repeat(pad_count, 1), action_hist], dim=0)
    return state_hist, action_hist, commands[t, env_id], padded, valid_len


def _rollout_batch(
    *,
    dynamics: Any,
    agent: Any,
    state_history: torch.Tensor,
    action_history: torch.Tensor,
    commands: torch.Tensor,
    horizon: int,
    action_temperature: float,
    device: torch.device,
    step_dt: float,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    batch = int(state_history.shape[0])
    full_action_dim = int(getattr(dynamics.cfg, "full_action_dim", dynamics.cfg.action_dim))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    model_ids = torch.randint(0, dynamics.ensemble_size, (batch,), generator=generator, device=device)
    reward_state = Go2RWMRewardState.create(
        num_envs=batch,
        action_dim=full_action_dim,
        device=device,
        step_dt=float(step_dt),
    )
    reward_state.last_joint_vel = state_history[:, -1, 21:33].clone()
    reward_state.last_action = action_history[:, -1].clone()

    rows = {key: [] for key in ("states", "actions", "next_states", "contacts", "terminations", "commands", "rewards", "prev_actions")}
    prev_action = action_history[:, -1].clone()
    for rollout_step in range(int(horizon)):
        state = state_history[:, -1].clone()
        full_obs = make_go2_policy_obs(state, commands, prev_action)
        actions_np = agent.sample_actions(
            interaction_step=0,
            prev_transition={"next_observation": full_obs.detach().cpu().numpy().astype(np.float32)},
            training=False,
            action_temperature=action_temperature,
        ).astype(np.float32)
        action = torch.as_tensor(actions_np, device=device, dtype=torch.float32)
        action_history = torch.cat([action_history[:, 1:], action.unsqueeze(1)], dim=1)
        with torch.no_grad():
            next_state, _aleatoric, epistemic, contact_logits, term_logits = dynamics.predict(
                state_history,
                action_history,
                model_ids,
            )
        contacts = (torch.sigmoid(contact_logits) > 0.5).float()
        rewards, _terms = compute_go2_imagination_reward(
            state=next_state,
            action=action,
            command=commands,
            foot_contact=contacts,
            # Keep the gait phase aligned with build_trace_replay_v4, which
            # recomputes candidate rewards using the rollout timestep.  A
            # constant zero here makes every RWM-reference step use phase 0
            # and corrupts candidate-minus-reference return features.
            episode_length=torch.full(
                (batch,), rollout_step, dtype=torch.long, device=device
            ),
            reward_state=reward_state,
            epistemic_uncertainty=epistemic,
        )
        terminations = (torch.sigmoid(term_logits).squeeze(-1) > 0.5) | bad_orientation_from_state(next_state)
        rows["states"].append(state.detach().cpu())
        rows["actions"].append(action.detach().cpu())
        rows["next_states"].append(next_state.detach().cpu())
        rows["contacts"].append(contacts.detach().cpu())
        rows["terminations"].append(terminations.detach().cpu())
        rows["commands"].append(commands.detach().cpu())
        rows["rewards"].append(rewards.detach().cpu())
        rows["prev_actions"].append(prev_action.detach().cpu())
        state_history = torch.cat([state_history[:, 1:], next_state.unsqueeze(1)], dim=1)
        prev_action = action

    stacked = {key: torch.stack(values, dim=0) for key, values in rows.items()}
    trajectories: list[dict[str, torch.Tensor]] = []
    for env_id in range(batch):
        term = stacked["terminations"][:, env_id].bool()
        stop = int(term.nonzero(as_tuple=False)[0].item()) + 1 if bool(term.any()) else int(horizon)
        trajectories.append({key: value[:stop, env_id] for key, value in stacked.items()})
    return trajectories


def main() -> None:
    args = _parse_args()
    protocol, _, _ = load_protocol(args.protocol)
    if float(args.action_temperature) != float(protocol["candidate"]["action_temperature"]):
        raise ValueError("RWM reference action temperature differs from TRACE V10 protocol.")
    if int(args.horizon) != int(protocol["candidate"]["horizon"]):
        raise ValueError("RWM reference horizon differs from TRACE V10 protocol.")
    if args.horizon < 1 or args.batch_size < 1:
        raise ValueError("horizon and batch_size must be positive.")
    configure_low_thread_env()
    device_name = select_device(args.device)
    device = torch.device(device_name)
    set_seed(int(args.seed))

    candidate_path = resolve_repo_path(args.candidate_dataset)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    source_path = _resolve_source_dataset(candidate, args.source_dataset)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    dynamics, checkpoint = load_any_go2_dynamics_checkpoint(resolve_repo_path(args.model_path), device=device)
    dynamics.eval()
    action_dim = int(getattr(dynamics.cfg, "full_action_dim", dynamics.cfg.action_dim))
    policy = _load_policy(resolve_repo_path(args.policy_path), num_envs=args.batch_size, action_dim=action_dim, device=device_name)
    source_data = _source_tensors(source)
    history_horizon = int(dynamics.cfg.history_horizon)
    step_dt = float((source.get("metadata") or {}).get("dt", 0.02))

    start_ids_raw = candidate.get("trace_start_state_ids")
    if start_ids_raw is None:
        raise ValueError("Candidate dataset is missing trace_start_state_ids.")
    start_ids = sorted({int(value) for value in torch.as_tensor(start_ids_raw).reshape(-1).tolist()})
    namespace = sha256_path(candidate_path)[:16]
    source_namespace = sha256_path(source_path)[:16]
    summaries: list[dict[str, Any]] = []
    for batch_start in range(0, len(start_ids), int(args.batch_size)):
        ids = start_ids[batch_start : batch_start + int(args.batch_size)]
        histories = [
            _history_for_flat_id(
                source_data,
                flat_id,
                history_horizon,
                allow_padded_history=bool(args.allow_padded_history),
            )
            for flat_id in ids
        ]
        state_history = torch.stack([item[0] for item in histories], dim=0).to(device)
        action_history = torch.stack([item[1] for item in histories], dim=0).to(device)
        commands = torch.stack([item[2] for item in histories], dim=0).to(device)
        if not hasattr(policy, "reset_action_noise"):
            raise RuntimeError("FlashSAC policy does not expose reset_action_noise().")
        policy.reset_action_noise((len(ids),))
        trajectories = _rollout_batch(
            dynamics=dynamics,
            agent=policy,
            state_history=state_history,
            action_history=action_history,
            commands=commands,
            horizon=int(args.horizon),
            action_temperature=float(args.action_temperature),
            device=device,
            step_dt=step_dt,
            seed=int(args.seed) + batch_start,
        )
        for flat_id, trajectory, history_info in zip(ids, trajectories, histories, strict=True):
            trajectory["rewards"] = _apply_replay_safety_costs(
                trajectory["rewards"],
                trajectory["actions"],
                trajectory["prev_actions"],
                trajectory["terminations"],
                terminal_penalty=float(args.terminal_penalty),
                override_terminal_reward=bool(args.override_terminal_reward),
                failure_backprop_steps=int(args.failure_backprop_steps),
                failure_backprop_penalty=float(args.failure_backprop_penalty),
                action_saturation_threshold=float(args.action_saturation_threshold),
                action_saturation_penalty_scale=float(args.action_saturation_penalty_scale),
                action_delta_penalty_scale=float(args.action_delta_penalty_scale),
            )
            trajectory["trajectory_id"] = f"rwm_ref_start{flat_id:08d}"
            trajectory["start_state_id"] = int(flat_id)
            trajectory["reset_reconstruction_error"] = 0.0
            trajectory["simulator_mismatch"] = {
                "source": "frozen_rwm_reference",
                "model_path": str(resolve_repo_path(args.model_path)),
                "policy_path": str(resolve_repo_path(args.policy_path)),
            }
            trajectory["nominal_horizon"] = int(args.horizon)
            summary = summarize_go2_trajectory(trajectory)
            _add_action_safety_summary(summary, trajectory, float(args.action_saturation_threshold))
            summary["start_state_key"] = f"{namespace}:{flat_id}"
            summary["source_state_key"] = f"{source_namespace}:{flat_id}"
            summary["comparison_group_key"] = summary["source_state_key"]
            summary["candidate_namespace"] = namespace
            summary["source_kind"] = "rwm_reference_rollout"
            summary["reference_history_padded"] = bool(history_info[3])
            summary["reference_history_valid_length"] = int(history_info[4])
            summaries.append(summary)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "candidate_dataset": str(candidate_path),
                "candidate_dataset_sha256": sha256_path(candidate_path),
                "source_dataset": str(source_path),
                "source_dataset_sha256": sha256_path(source_path),
                "model_path": str(resolve_repo_path(args.model_path)),
                "policy_path": str(resolve_repo_path(args.policy_path)),
                "history_horizon": history_horizon,
                "history_action_alignment": "source_prev_actions_then_policy_action",
                "horizon": int(args.horizon),
                "summary_count": len(summaries),
                "output": str(output),
                "model_infos": dict(checkpoint.get("infos") or {}),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
