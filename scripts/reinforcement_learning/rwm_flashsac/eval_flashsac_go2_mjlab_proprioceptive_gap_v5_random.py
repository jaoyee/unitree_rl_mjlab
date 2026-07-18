"""Evaluate Go2 FlashSAC-RWM policies with command-response safety gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac.agent_proprioceptive import (
    create_go2_flashsac_proprioceptive_agent,
)
from scripts.reinforcement_learning.rwm_flashsac.command_metrics_v4 import (
    summarize_command_segments,
)
from scripts.reinforcement_learning.rwm_dataset.broken_go2 import (
    apply_go2_pd_joint_strength_scales,
)
from scripts.reinforcement_learning.rwm_dataset.go2_contact_velocity_estimator import (
    estimate_go2_base_lin_vel_b,
)
from flash_rl.envs.mjlab import (
    configure_mjlab_randomization,
    normalize_friend_flat_dr_components,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    apply_policy_observation_mask_np,
    configure_low_thread_env,
    expand_policy_actions_np,
    get_world_model_broken_joint_names,
    get_world_model_action_masks,
    get_world_model_policy_observation_mask,
    load_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    scalarize,
    select_device,
    set_seed,
)
from scripts.reinforcement_learning.rwm_flashsac.world_model_env_proprioceptive import (
    CRITIC_OBS_DIM_FULL_RWM,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--task", default="Unitree-Go2-Flat-RWM-Pretrain-Ens")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fixed_command", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=None)
    parser.add_argument(
        "--command_sequence",
        action="append",
        default=None,
        help=(
            "Repeat for each vx,vy,yaw command; use --command_sequence=value "
            "when the first component is negative."
        ),
    )
    parser.add_argument("--command_switch_steps", type=int, default=300)
    parser.add_argument(
        "--random_command_schedule_path",
        default=None,
        help="NPZ containing commands[steps,num_envs,3]; overrides the environment command sampler.",
    )
    parser.add_argument("--command_settle_steps", type=int, default=50)
    parser.add_argument("--sustained_response_steps", type=int, default=10)
    parser.add_argument("--xy_error_threshold", type=float, default=0.15)
    parser.add_argument("--yaw_error_threshold", type=float, default=0.15)
    parser.add_argument("--stand_xy_speed_threshold", type=float, default=0.10)
    parser.add_argument("--stand_yaw_speed_threshold", type=float, default=0.10)
    parser.add_argument("--minimum_response_gain", type=float, default=0.50)
    parser.add_argument("--maximum_response_gain", type=float, default=1.50)
    parser.add_argument("--no_response_gain", type=float, default=0.20)
    parser.add_argument("--action_saturation_threshold", type=float, default=0.95)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--randomization_preset",
        choices=(
            "default",
            "friend_flat",
            "calibrated_default",
            "calibrated_friend_flat",
        ),
        default="default",
        help="Physical/observation DR preset used with --no-clean.",
    )
    parser.add_argument(
        "--randomization_components",
        default="all",
        help=(
            "Comma/space-separated friend_flat components used with --no-clean. "
            "Allowed: friction,mass_com,motor,delay,observation,push,initial_state,all."
        ),
    )
    parser.add_argument(
        "--randomization_scale",
        type=float,
        default=1.0,
        help="Scale friend_flat ranges around nominal centers; must be in [0, 1].",
    )
    parser.add_argument(
        "--payload_mass_kg",
        type=float,
        default=0.0,
        help="Fixed payload mass added above the base link for gap evaluation.",
    )
    parser.add_argument(
        "--rr_calf_strength",
        type=float,
        default=1.0,
        help="Fixed RR calf motor-strength scale for gap evaluation.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--validate_contact_velocity_estimator",
        action="store_true",
        help="Compare the deployment-observable contact estimator with simulator base velocity truth.",
    )
    parser.add_argument(
        "--velocity_estimator_samples_npz",
        default=None,
        help="Optional NPZ output containing per-step estimator inputs and simulator truth.",
    )
    parser.add_argument("--broken_joint_names", nargs="*", default=None)
    parser.add_argument(
        "--joint_strength_scales",
        nargs="*",
        default=(),
        metavar="JOINT=SCALE",
        help="Per-joint actuator strength scales for mjlab eval, e.g. RR_calf_joint=0.5.",
    )
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _disable_randomization(env_cfg: Any) -> None:
    env_cfg.events.pop("push_robot", None)
    for event_name in ("foot_friction", "encoder_bias", "base_com"):
        env_cfg.events.pop(event_name, None)
    for group_name in ("actor", "critic"):
        obs_group = env_cfg.observations.get(group_name)
        if obs_group is not None:
            obs_group.enable_corruption = False
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = {}
    if hasattr(env_cfg, "curriculums"):
        env_cfg.curriculums = {}


def _actor_obs(obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
    return obs_dict["actor"].detach().cpu().numpy().astype(np.float32)


def _parse_joint_strength_scales(spec: list[str] | tuple[str, ...]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for item in spec:
        raw = str(item).strip()
        if not raw:
            continue
        if "=" in raw:
            name, value = raw.split("=", maxsplit=1)
        elif ":" in raw:
            name, value = raw.split(":", maxsplit=1)
        else:
            raise ValueError(f"Invalid joint strength scale {raw!r}; expected JOINT=SCALE.")
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid joint strength scale {raw!r}; joint name is empty.")
        scale = float(value)
        if scale < 0.0 or scale > 1.0:
            raise ValueError(f"Joint strength scale for {name!r} must be in [0, 1], got {scale}.")
        scales[name] = scale
    return scales


def _parse_command_sequence(values: list[str] | None) -> list[tuple[float, float, float]]:
    commands: list[tuple[float, float, float]] = []
    for value in values or []:
        fields = value.replace(";", ",").split(",")
        if len(fields) != 3:
            raise ValueError(f"Expected command triplet vx,vy,yaw, got {value!r}")
        commands.append(tuple(float(field) for field in fields))
    return commands


def _configure_fixed_command_range(
    env_cfg: Any,
    fixed_command: tuple[float, float, float] | None,
    command_sequence: list[tuple[float, float, float]],
    random_command_schedule: np.ndarray | None = None,
) -> None:
    commands = command_sequence or ([fixed_command] if fixed_command is not None else [])
    if random_command_schedule is not None:
        maxima = np.max(np.abs(random_command_schedule), axis=(0, 1))
        commands = [tuple(float(value) for value in maxima)]
    if not commands or "twist" not in env_cfg.commands:
        return
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.ranges.lin_vel_x = (-max(0.1, max(abs(cmd[0]) for cmd in commands)), max(0.1, max(abs(cmd[0]) for cmd in commands)))
    twist_cmd.ranges.lin_vel_y = (-max(0.1, max(abs(cmd[1]) for cmd in commands)), max(0.1, max(abs(cmd[1]) for cmd in commands)))
    twist_cmd.ranges.ang_vel_z = (-max(0.1, max(abs(cmd[2]) for cmd in commands)), max(0.1, max(abs(cmd[2]) for cmd in commands)))
    if hasattr(twist_cmd.ranges, "heading"):
        twist_cmd.ranges.heading = None
    if hasattr(twist_cmd, "heading_command"):
        twist_cmd.heading_command = False
    if hasattr(twist_cmd, "rel_standing_envs"):
        twist_cmd.rel_standing_envs = 0.0
    if hasattr(twist_cmd, "rel_heading_envs"):
        twist_cmd.rel_heading_envs = 0.0
    if hasattr(twist_cmd, "init_velocity_prob"):
        twist_cmd.init_velocity_prob = 0.0


def _force_fixed_command(env: Any, fixed_command: tuple[float, float, float] | np.ndarray | None) -> None:
    if fixed_command is None:
        return
    try:
        command = env.unwrapped.command_manager.get_term("twist")
    except Exception:
        return
    value = torch.as_tensor(fixed_command, dtype=torch.float32, device=env.unwrapped.device)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(command.vel_command_b.shape[0], -1)
    if value.shape != command.vel_command_b.shape:
        raise ValueError(
            f"Forced command shape mismatch: expected {tuple(command.vel_command_b.shape)}, "
            f"got {tuple(value.shape)}."
        )
    if hasattr(command, "vel_command_b"):
        command.vel_command_b[:, :] = value
    if hasattr(command, "is_standing_env"):
        command.is_standing_env[:] = torch.linalg.vector_norm(value, dim=-1) <= 1.0e-8
    if hasattr(command, "is_heading_env"):
        command.is_heading_env[:] = False


def _apply_command_to_full_obs(
    observations: np.ndarray,
    fixed_command: tuple[float, float, float] | np.ndarray | None,
) -> np.ndarray:
    if fixed_command is None:
        return observations
    observations = observations.copy()
    value = np.asarray(fixed_command, dtype=np.float32)
    if value.ndim == 1:
        value = np.broadcast_to(value, (observations.shape[0], 3))
    if value.shape != (observations.shape[0], 3):
        raise ValueError(
            f"Observation command shape mismatch: expected {(observations.shape[0], 3)}, got {value.shape}."
        )
    observations[:, 9:12] = value
    return observations


def _mean(values: list[float], fallback: float = 0.0) -> float:
    return float(np.mean(values)) if values else fallback


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse_args()

    checkpoint_path = resolve_repo_path(args.checkpoint_path)
    config_path = Path(args.config_path).expanduser() if args.config_path else checkpoint_path / "rwm_flashsac_config.yaml"
    if not config_path.is_absolute():
        config_path = resolve_repo_path(config_path)
    cfg = load_config(config_path, overrides=args.overrides)

    device = select_device(args.device or cfg.agent.device_type)
    set_seed(args.seed)
    broken_joint_names = get_world_model_broken_joint_names(cfg, args.broken_joint_names)
    joint_strength_scales = _parse_joint_strength_scales(args.joint_strength_scales)
    command_sequence = _parse_command_sequence(args.command_sequence)
    random_command_schedule: np.ndarray | None = None
    random_command_schedule_sha256: str | None = None
    if args.random_command_schedule_path:
        if args.fixed_command is not None or command_sequence:
            raise ValueError(
                "--random_command_schedule_path cannot be combined with fixed/sequence commands."
            )
        schedule_path = Path(args.random_command_schedule_path).expanduser()
        if not schedule_path.is_absolute():
            schedule_path = resolve_repo_path(schedule_path)
        with np.load(schedule_path) as payload:
            random_command_schedule = np.asarray(payload["commands"], dtype=np.float32)
        expected = (args.steps, args.num_envs, 3)
        if random_command_schedule.shape != expected:
            raise ValueError(
                f"Random command schedule must have shape {expected}, "
                f"got {random_command_schedule.shape}."
            )
        random_command_schedule_sha256 = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    randomization_components = (
        frozenset()
        if args.clean
        else normalize_friend_flat_dr_components(args.randomization_components)
    )

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    # The E/D experiment is flat-ground only. Some play configs retain a
    # terrain-reset event even though terrain is not an experimental factor.
    env_cfg.events.pop("randomize_terrain", None)

    # Match dataset collection ordering: establish the weakened/broken robot
    # first, then wrap those actuators with the selected DR preset.
    effective_joint_strength_scales = dict(joint_strength_scales)
    for joint_name in broken_joint_names:
        effective_joint_strength_scales[joint_name] = 0.0
    if effective_joint_strength_scales:
        apply_go2_pd_joint_strength_scales(env_cfg, effective_joint_strength_scales)

    gap_kwargs = {
        "payload_mass_range_kg": (args.payload_mass_kg, args.payload_mass_kg),
        "payload_position_body_m": (0.0, 0.0, 0.10),
        "payload_box_size_m": (0.20, 0.12, 0.05),
        "rr_calf_strength_range": (args.rr_calf_strength, args.rr_calf_strength),
    }
    if args.clean:
        _disable_randomization(env_cfg)
        configure_mjlab_randomization(
            env_cfg,
            use_domain_randomization=False,
            use_push_randomization=False,
            use_observation_noise=False,
            **gap_kwargs,
        )
    else:
        configure_mjlab_randomization(
            env_cfg,
            use_domain_randomization=True,
            use_push_randomization=True,
            use_observation_noise=True,
            randomization_preset=args.randomization_preset,
            randomization_components=randomization_components,
            randomization_scale=args.randomization_scale,
            **gap_kwargs,
        )
    fixed_command = tuple(args.fixed_command) if args.fixed_command is not None else None
    _configure_fixed_command_range(
        env_cfg, fixed_command, command_sequence, random_command_schedule
    )
    initial_command = (
        random_command_schedule[0]
        if random_command_schedule is not None
        else (command_sequence[0] if command_sequence else fixed_command)
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _force_fixed_command(env, initial_command)
    full_actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    full_action_dim = int(env.single_action_space.shape[0])
    if full_actor_dim != 48:
        raise RuntimeError(f"Expected 48-dim RWM actor observation, got {full_actor_dim}.")

    policy_action_mask_indices, world_model_action_mask_indices = get_world_model_action_masks(cfg, full_action_dim)
    policy_observation_mask_indices = get_world_model_policy_observation_mask(cfg, full_actor_dim)
    policy_action_dim = full_action_dim - len(policy_action_mask_indices)
    policy_observation_dim = full_actor_dim - len(policy_observation_mask_indices)

    obs_space, action_space = make_vector_spaces(
        args.num_envs,
        obs_dim=policy_observation_dim,
        action_dim=policy_action_dim,
    )
    agent_cfg = make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_proprioceptive_agent(obs_space, action_space, agent_cfg)
    agent.load(str(checkpoint_path))

    obs_dict, _ = env.reset()
    _force_fixed_command(env, initial_command)
    full_observations = _apply_command_to_full_obs(_actor_obs(obs_dict), initial_command)

    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    total_reward_sum = 0.0
    total_reward_samples = 0
    ever_terminated = np.zeros(args.num_envs, dtype=bool)
    first_termination_step = np.full(args.num_envs, -1, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    terminated_count = 0
    timeout_count = 0
    logged_metrics: dict[str, list[float]] = defaultdict(list)
    base_lin_vel_samples: list[np.ndarray] = []
    base_ang_vel_samples: list[np.ndarray] = []
    command_samples: list[np.ndarray] = []
    action_abs_samples: list[float] = []
    action_samples: list[np.ndarray] = []
    terminated_samples: list[np.ndarray] = []
    truncated_samples: list[np.ndarray] = []
    estimated_base_lin_vel_samples: list[np.ndarray] = []
    estimated_base_lin_vel_truth_samples: list[np.ndarray] = []
    estimator_confidence_samples: list[np.ndarray] = []
    estimator_per_foot_samples: list[np.ndarray] = []
    estimator_contact_samples: list[np.ndarray] = []

    print(f"[Go2-FlashSAC-RWM-Proprioceptive-Eval] checkpoint={checkpoint_path}")
    print(
        "[Go2-FlashSAC-RWM-Proprioceptive-Eval] "
        f"task={args.task}, clean={args.clean}, randomization_preset={args.randomization_preset}, "
        f"randomization_components={sorted(randomization_components)}, "
        f"randomization_scale={args.randomization_scale}, "
        f"payload_mass_kg={args.payload_mass_kg}, "
        f"rr_calf_strength={args.rr_calf_strength}, "
        f"device={device}, num_envs={args.num_envs}"
    )
    print(f"[Go2-FlashSAC-RWM-Proprioceptive-Eval] broken_joint_names={list(broken_joint_names)}")
    print(f"[Go2-FlashSAC-RWM-Proprioceptive-Eval] joint_strength_scales={joint_strength_scales}")
    print(
        "[Go2-FlashSAC-RWM-Proprioceptive-Eval] "
        f"full_action_dim={full_action_dim}, policy_action_dim={policy_action_dim}, "
        f"policy_action_mask_indices={list(policy_action_mask_indices)}, "
        f"world_model_action_mask_indices={list(world_model_action_mask_indices)}, "
        f"policy_observation_mask_indices={list(policy_observation_mask_indices)}"
    )
    if fixed_command is not None:
        print(f"[Go2-FlashSAC-RWM-Proprioceptive-Eval] fixed_command={fixed_command}")
    if command_sequence:
        print(
            "[Go2-FlashSAC-RWM-Proprioceptive-Eval] "
            f"command_sequence={command_sequence}, switch_steps={args.command_switch_steps}"
        )

    with torch.no_grad():
        for _step in range(args.steps):
            current_command = (
                random_command_schedule[_step]
                if random_command_schedule is not None
                else (
                    command_sequence[(_step // max(1, args.command_switch_steps)) % len(command_sequence)]
                    if command_sequence
                    else fixed_command
                )
            )
            _force_fixed_command(env, current_command)
            full_observations = _apply_command_to_full_obs(full_observations, current_command)
            policy_observations = apply_policy_observation_mask_np(
                full_observations,
                policy_observation_mask_indices,
            )
            actions_np = agent.sample_actions(
                interaction_step=0,
                prev_transition={"next_observation": policy_observations},
                training=False,
            )
            actions_np = expand_policy_actions_np(
                actions_np,
                action_dim=full_action_dim,
                policy_action_mask_indices=policy_action_mask_indices,
                world_model_action_mask_indices=world_model_action_mask_indices,
            )
            action_abs_samples.append(float(np.abs(actions_np).mean()))
            action_samples.append(actions_np.copy())
            base_lin_vel_samples.append(full_observations[:, 0:3].copy())
            base_ang_vel_samples.append(full_observations[:, 3:6].copy())
            command_samples.append(full_observations[:, 9:12].copy())
            if args.validate_contact_velocity_estimator:
                robot_data = env.scene["robot"].data
                contact_data = env.scene["feet_ground_contact"].data.found
                if contact_data is None:
                    contact = torch.zeros(args.num_envs, 4, device=device)
                else:
                    contact = (contact_data > 0).float()
                estimated_velocity, confidence, per_foot_velocity = estimate_go2_base_lin_vel_b(
                    robot_data.joint_pos,
                    robot_data.joint_vel,
                    robot_data.root_link_ang_vel_b,
                    contact,
                )
                estimated_base_lin_vel_samples.append(
                    estimated_velocity.detach().cpu().numpy().astype(np.float32)
                )
                estimated_base_lin_vel_truth_samples.append(
                    robot_data.root_link_lin_vel_b.detach().cpu().numpy().astype(np.float32)
                )
                estimator_confidence_samples.append(
                    confidence.detach().cpu().numpy().astype(np.float32)
                )
                estimator_per_foot_samples.append(
                    per_foot_velocity.detach().cpu().numpy().astype(np.float32)
                )
                estimator_contact_samples.append(
                    contact.detach().cpu().numpy().astype(np.float32)
                )
            actions_t = torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)
            obs_dict, rewards, terminateds, truncateds, extras = env.step(actions_t)
            rewards_np = rewards.detach().cpu().numpy().astype(np.float64)
            term_np = terminateds.detach().cpu().numpy().astype(bool)
            trunc_np = truncateds.detach().cpu().numpy().astype(bool)
            done_np = np.logical_or(term_np, trunc_np)
            terminated_samples.append(term_np.copy())
            truncated_samples.append(trunc_np.copy())
            newly_terminated = term_np & (first_termination_step < 0)
            first_termination_step[newly_terminated] = _step + 1
            total_reward_sum += float(rewards_np.sum())
            total_reward_samples += int(rewards_np.size)
            ever_terminated |= term_np

            ep_returns += rewards_np
            ep_lengths += 1

            if done_np.any():
                completed_returns.extend(ep_returns[done_np].tolist())
                completed_lengths.extend(ep_lengths[done_np].tolist())
                terminated_count += int(term_np.sum())
                timeout_count += int(trunc_np.sum())
                ep_returns[done_np] = 0.0
                ep_lengths[done_np] = 0

            for key, value in (extras.get("log") or {}).items():
                logged_metrics[key].append(float(scalarize(value)))

            _force_fixed_command(env, current_command)
            full_observations = _apply_command_to_full_obs(_actor_obs(obs_dict), current_command)

    step_dt = float(env.step_dt)
    env.close()

    mean_return = _mean(completed_returns, fallback=float(ep_returns.mean()))
    std_return = float(np.std(completed_returns)) if completed_returns else float(np.std(ep_returns))
    mean_episode_length = _mean(completed_lengths, fallback=float(ep_lengths.mean()))
    base_lin_vel = np.concatenate(base_lin_vel_samples, axis=0) if base_lin_vel_samples else np.zeros((1, 3))
    base_ang_vel = np.concatenate(base_ang_vel_samples, axis=0) if base_ang_vel_samples else np.zeros((1, 3))
    commands = np.concatenate(command_samples, axis=0) if command_samples else np.zeros((1, 3))
    vel_error_xy = np.linalg.norm(base_lin_vel[:, 0:2] - commands[:, 0:2], axis=1)
    yaw_error = np.abs(base_ang_vel[:, 2] - commands[:, 2])
    restricted_first_termination_steps = np.where(
        first_termination_step >= 0, first_termination_step, args.steps
    )
    command_metrics = {}
    if command_sequence:
        command_metrics = summarize_command_segments(
            base_lin_vel=np.stack(base_lin_vel_samples, axis=0),
            base_yaw_vel=np.stack(base_ang_vel_samples, axis=0)[:, :, 2],
            commands=np.stack(command_samples, axis=0),
            terminated=np.stack(terminated_samples, axis=0),
            truncated=np.stack(truncated_samples, axis=0),
            actions=np.stack(action_samples, axis=0),
            command_sequence=command_sequence,
            command_switch_steps=args.command_switch_steps,
            step_dt=step_dt,
            settle_steps=args.command_settle_steps,
            sustained_response_steps=args.sustained_response_steps,
            xy_error_threshold=args.xy_error_threshold,
            yaw_error_threshold=args.yaw_error_threshold,
            stand_xy_speed_threshold=args.stand_xy_speed_threshold,
            stand_yaw_speed_threshold=args.stand_yaw_speed_threshold,
            minimum_response_gain=args.minimum_response_gain,
            maximum_response_gain=args.maximum_response_gain,
            no_response_gain=args.no_response_gain,
            action_saturation_threshold=args.action_saturation_threshold,
        )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "task": str(args.task),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "num_envs": int(args.num_envs),
        "clean": bool(args.clean),
        "randomization_preset": "clean" if args.clean else str(args.randomization_preset),
        "randomization_components": sorted(randomization_components),
        "randomization_scale": 0.0 if args.clean else float(args.randomization_scale),
        "payload_mass_kg": float(args.payload_mass_kg),
        "rr_calf_strength": float(args.rr_calf_strength),
        "joint_strength_scales": dict(effective_joint_strength_scales),
        "command_sequence": [list(command) for command in command_sequence],
        "command_switch_steps": int(args.command_switch_steps),
        "random_command_schedule_path": args.random_command_schedule_path,
        "random_command_schedule_sha256": random_command_schedule_sha256,
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_episode_length": mean_episode_length,
        "global_mean_step_reward": (
            total_reward_sum / total_reward_samples if total_reward_samples else 0.0
        ),
        "horizon_survival_fraction": float((~ever_terminated).mean()),
        "ever_terminated_fraction": float(ever_terminated.mean()),
        "restricted_mean_first_termination_step": float(
            restricted_first_termination_steps.mean()
        ),
        "median_restricted_first_termination_step": float(
            np.median(restricted_first_termination_steps)
        ),
        "completed_episodes": len(completed_returns),
        "terminated_count": terminated_count,
        "timeout_count": timeout_count,
        "non_timeout_termination_count": terminated_count,
        "command_x": float(commands[:, 0].mean()),
        "command_y": float(commands[:, 1].mean()),
        "command_yaw": float(commands[:, 2].mean()),
        "base_lin_vel_x": float(base_lin_vel[:, 0].mean()),
        "base_lin_vel_y": float(base_lin_vel[:, 1].mean()),
        "base_speed_xy": float(np.linalg.norm(base_lin_vel[:, 0:2], axis=1).mean()),
        "base_yaw_vel": float(base_ang_vel[:, 2].mean()),
        "error_vel_xy": float(vel_error_xy.mean()),
        "error_vel_yaw": float(yaw_error.mean()),
        "action_abs_mean": _mean(action_abs_samples),
        "command_metrics_v4": command_metrics,
    }

    if args.validate_contact_velocity_estimator and estimated_base_lin_vel_samples:
        estimated = np.concatenate(estimated_base_lin_vel_samples, axis=0)
        estimated_truth = np.concatenate(estimated_base_lin_vel_truth_samples, axis=0)
        confidence = np.concatenate(estimator_confidence_samples, axis=0)
        valid = confidence > 0.0
        error = estimated[valid] - estimated_truth[valid]
        if error.size:
            summary["contact_velocity_estimator"] = {
                "valid_fraction": float(valid.mean()),
                "mean_confidence": float(confidence.mean()),
                "mae_xyz": np.mean(np.abs(error), axis=0).tolist(),
                "rmse_xyz": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
                "mae_xy": float(np.linalg.norm(error[:, :2], axis=1).mean()),
                "rmse_xy": float(np.sqrt(np.mean(np.sum(np.square(error[:, :2]), axis=1)))),
                "truth_speed_xy_mean": float(
                    np.linalg.norm(estimated_truth[valid, :2], axis=1).mean()
                ),
                "estimated_speed_xy_mean": float(
                    np.linalg.norm(estimated[valid, :2], axis=1).mean()
                ),
            }
        if args.velocity_estimator_samples_npz:
            samples_path = Path(args.velocity_estimator_samples_npz)
            if not samples_path.is_absolute():
                samples_path = resolve_repo_path(samples_path)
            samples_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                samples_path,
                estimated=np.stack(estimated_base_lin_vel_samples, axis=0),
                truth=np.stack(estimated_base_lin_vel_truth_samples, axis=0),
                confidence=np.stack(estimator_confidence_samples, axis=0),
                per_foot=np.stack(estimator_per_foot_samples, axis=0),
                contact=np.stack(estimator_contact_samples, axis=0),
            )
            summary["contact_velocity_estimator"]["samples_npz"] = str(samples_path)

    print("[Go2-FlashSAC-RWM-Proprioceptive-Eval] Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    key_metrics = (
        "Episode_Termination/fell_over",
        "Episode_Termination/illegal_contact",
        "Metrics/twist/error_vel_xy",
        "Metrics/twist/error_vel_yaw",
    )
    for key in key_metrics:
        values = logged_metrics.get(key)
        if values:
            summary[key] = float(np.mean(values))
            print(f"  {key}: {summary[key]:.6f}")

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = resolve_repo_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Go2-FlashSAC-RWM-Proprioceptive-Eval] wrote {output_path}")


if __name__ == "__main__":
    main()
