#!/usr/bin/env python3
"""Validate Go2 TRACE reset by replaying dataset actions in the source environment."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.envs.mjlab import configure_mjlab_randomization
from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    select_device,
    set_seed,
)
from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
    SNAPSHOT_VERSION,
    restore_go2_simulator_snapshot,
    step_without_automatic_reset,
)
from src.tasks.rwm_velocity.mdp.extractors import Go2RWMExtractor


STATE_BLOCKS = {
    "base_lin_vel": (0, 3),
    "base_ang_vel": (3, 6),
    "projected_gravity": (6, 9),
    "joint_pos": (9, 21),
    "joint_vel": (21, 33),
    "actuator_force": (33, 45),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--task",
        default="Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9102)
    parser.add_argument("--num-windows", type=int, default=64)
    parser.add_argument(
        "--min-source-timestep",
        type=int,
        default=1,
        help="Exclude episode-reset rows whose logged derived actuator force may precede reset finalization.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 5, 10))
    parser.add_argument("--payload-mass-kg", type=float, default=0.0)
    parser.add_argument("--rr-calf-strength", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=5.0e-3)
    return parser.parse_args()


def flattened(dataset: dict[str, Any], key: str, width: int | None = None) -> torch.Tensor:
    value = stack_time_key(dataset, key)
    if width is None:
        return value.reshape(-1)
    return value.reshape(-1, width)


def force_commands(env: Any, command: torch.Tensor, env_ids: torch.Tensor) -> None:
    term = env.command_manager.get_term("twist")
    command = command.to(device=env.device, dtype=torch.float32)
    if hasattr(term, "vel_command_b"):
        term.vel_command_b[env_ids] = command
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[env_ids] = torch.linalg.norm(command, dim=-1) < 1.0e-8
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[env_ids] = False


def error_summary(error: torch.Tensor) -> dict[str, float]:
    error = error.detach().float().cpu().reshape(-1)
    return {
        "mean": float(error.mean()),
        "median": float(error.median()),
        "p95": float(torch.quantile(error, 0.95)),
        "max": float(error.max()),
    }


def state_error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    absolute = torch.abs(actual - expected)
    result: dict[str, Any] = {
        "l_inf_per_window": error_summary(absolute.max(dim=-1).values),
        "l2_per_window": error_summary(torch.linalg.norm(actual - expected, dim=-1)),
        "blocks": {},
    }
    for name, (start, stop) in STATE_BLOCKS.items():
        block = absolute[:, start:stop]
        result["blocks"][name] = {
            "abs_mean": float(block.mean()),
            "abs_max": float(block.max()),
        }
    return result


def select_windows(
    episode_ids: torch.Tensor,
    timesteps: torch.Tensor,
    dones: torch.Tensor,
    horizon: int,
    count: int,
    seed: int,
    min_timestep: int,
) -> list[list[int]]:
    row_by_key = {
        (int(episode), int(timestep)): index
        for index, (episode, timestep) in enumerate(zip(episode_ids.tolist(), timesteps.tolist()))
    }
    windows: list[list[int]] = []
    for start, (episode, timestep) in enumerate(zip(episode_ids.tolist(), timesteps.tolist())):
        if int(timestep) < int(min_timestep):
            continue
        indices = [row_by_key.get((int(episode), int(timestep) + offset)) for offset in range(horizon)]
        if any(index is None for index in indices):
            continue
        rows = [int(index) for index in indices if index is not None]
        if bool(dones[rows[:-1]].any()) if horizon > 1 else False:
            continue
        windows.append(rows)
    if len(windows) < count:
        raise RuntimeError(f"Only {len(windows)} contiguous {horizon}-step windows are available, need {count}.")
    generator = random.Random(seed)
    generator.shuffle(windows)
    return windows[:count]


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = parse_args()
    if not args.horizons or min(args.horizons) < 1:
        raise ValueError("horizons must be positive")
    max_horizon = max(args.horizons)
    device = select_device(args.device)
    set_seed(args.seed)

    dataset_path = Path(args.dataset).resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    states = flattened(dataset, "states", 45).float()
    next_states = flattened(dataset, "next_states", 45).float()
    actions = flattened(dataset, "actions", 12).float()
    commands = flattened(dataset, "commands", 3).float()
    episode_ids = flattened(dataset, "episode_ids").long()
    timesteps = flattened(dataset, "timesteps").long()
    dones = flattened(dataset, "dones").bool()
    contacts = flattened(dataset, "contacts", 4).float() if "contacts" in dataset else None
    terminations = (
        flattened(dataset, "terminations", 1).reshape(-1).bool()
        if "terminations" in dataset
        else dones
    )
    rewards = flattened(dataset, "rewards").float() if "rewards" in dataset else None
    windows = select_windows(
        episode_ids,
        timesteps,
        dones,
        max_horizon,
        args.num_windows,
        args.seed,
        args.min_source_timestep,
    )
    start_rows = torch.tensor([window[0] for window in windows], dtype=torch.long)

    snapshot_keys = {
        "root_state_local": ("sim_root_states_local", 13),
        "joint_position": ("sim_joint_positions", 12),
        "joint_velocity": ("sim_joint_velocities", 12),
        "action": ("sim_action_histories", 12),
        "prev_action": ("sim_prev_action_histories", 12),
        "prev_prev_action": ("sim_prev_prev_action_histories", 12),
        "command": ("sim_snapshot_commands", 3),
    }
    snapshot: dict[str, Any] = {"snapshot_version": SNAPSHOT_VERSION}
    for target_key, (dataset_key, width) in snapshot_keys.items():
        if dataset_key not in dataset:
            raise KeyError(f"Dataset is missing required snapshot field {dataset_key!r}")
        snapshot[target_key] = flattened(dataset, dataset_key, width)[start_rows]
    # Match the collector's authoritative source-action convention.
    snapshot["action"] = flattened(dataset, "prev_actions", 12)[start_rows]

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_windows
    env_cfg.seed = args.seed
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=False,
        use_push_randomization=False,
        use_observation_noise=False,
        randomization_preset="default",
        randomization_components=None,
        randomization_scale=1.0,
        payload_mass_range_kg=(args.payload_mass_kg, args.payload_mass_kg),
        payload_position_body_m=(0.0, 0.0, 0.1),
        payload_box_size_m=(0.2, 0.12, 0.05),
        rr_calf_strength_range=(args.rr_calf_strength, args.rr_calf_strength),
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    extractor = Go2RWMExtractor(env.unwrapped)
    env.reset()
    env_ids = torch.arange(args.num_windows, device=env.unwrapped.device)
    snapshot = {
        key: value.to(env.unwrapped.device) if isinstance(value, torch.Tensor) else value
        for key, value in snapshot.items()
    }
    restore_go2_simulator_snapshot(env.unwrapped, snapshot, env_ids)
    reset_actual = extractor.extract_state().clone()
    reset_expected = states[start_rows].to(env.unwrapped.device)

    report: dict[str, Any] = {
        "format_version": "go2_trace_source_reset_validation_v1",
        "dataset": str(dataset_path),
        "dataset_sha256": None,
        "source_environment": {
            "payload_mass_kg": args.payload_mass_kg,
            "rr_calf_strength": args.rr_calf_strength,
            "domain_randomization": False,
            "push_randomization": False,
            "observation_noise": False,
        },
        "num_windows": args.num_windows,
        "min_source_timestep": args.min_source_timestep,
        "max_horizon": max_horizon,
        "start_rows": start_rows.tolist(),
        "reset": state_error_summary(reset_actual, reset_expected),
        "horizons": {},
    }
    reset_absolute = torch.abs(reset_actual - reset_expected)
    reset_window_max, reset_dimension = reset_absolute.max(dim=-1)
    worst_count = min(8, args.num_windows)
    worst_values, worst_indices = torch.topk(reset_window_max, k=worst_count)
    report["reset_worst_windows"] = [
        {
            "source_row": int(start_rows[index].item()),
            "window_index": int(index),
            "max_abs_error": float(value),
            "state_dimension": int(reset_dimension[index].item()),
            "expected_actuator_force": reset_expected[index, 33:45].detach().cpu().tolist(),
            "actual_actuator_force": reset_actual[index, 33:45].detach().cpu().tolist(),
            "snapshot_action": snapshot["action"][index].detach().cpu().tolist(),
        }
        for value, index in zip(worst_values, worst_indices)
    ]

    for step in range(1, max_horizon + 1):
        rows = torch.tensor([window[step - 1] for window in windows], dtype=torch.long)
        force_commands(env.unwrapped, commands[rows].to(env.unwrapped.device), env_ids)
        _, actual_reward, actual_termination, actual_timeout, _ = step_without_automatic_reset(
            env.unwrapped,
            actions[rows].to(env.unwrapped.device),
        )
        actual_state = extractor.extract_state().clone()
        expected_state = next_states[rows].to(env.unwrapped.device)
        if step in args.horizons:
            item: dict[str, Any] = state_error_summary(actual_state, expected_state)
            item["termination_mismatch_rate"] = float(
                (actual_termination.bool() != terminations[rows].to(env.unwrapped.device)).float().mean()
            )
            item["timeout_rate"] = float(actual_timeout.float().mean())
            if contacts is not None:
                actual_contact = extractor.extract_contact().float()
                item["contact_mismatch_rate"] = float(
                    (actual_contact.bool() != contacts[rows].to(env.unwrapped.device).bool()).float().mean()
                )
            if rewards is not None:
                item["reward_abs_error"] = error_summary(
                    torch.abs(actual_reward.float() - rewards[rows].to(env.unwrapped.device))
                )
            report["horizons"][str(step)] = item

    reset_max = report["reset"]["l_inf_per_window"]["max"]
    reset_kinematic_max = max(
        report["reset"]["blocks"][name]["abs_max"]
        for name in STATE_BLOCKS
        if name != "actuator_force"
    )
    one_step_max = report["horizons"]["1"]["l_inf_per_window"]["max"]
    report["gate"] = {
        "tolerance": args.tolerance,
        "reset_full_state_passed": bool(reset_max <= args.tolerance),
        "reset_kinematic_state_passed": bool(reset_kinematic_max <= args.tolerance),
        "one_step_passed": bool(one_step_max <= args.tolerance),
        "transition_replay_passed": bool(one_step_max <= args.tolerance),
        "passed": bool(reset_max <= args.tolerance and one_step_max <= args.tolerance),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    env.close()
    if not report["gate"]["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
