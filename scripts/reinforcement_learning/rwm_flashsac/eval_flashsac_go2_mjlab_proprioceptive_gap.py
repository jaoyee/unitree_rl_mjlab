"""Evaluate proprioceptive-lin-vel Go2 FlashSAC-RWM policy in mjlab."""

from __future__ import annotations

import argparse
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
from scripts.reinforcement_learning.rwm_flashsac.dynamics_loader import (
    load_any_go2_dynamics_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional frozen RWM checkpoint used to score epistemic uncertainty on true-env transitions.",
    )
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
) -> None:
    commands = command_sequence or ([fixed_command] if fixed_command is not None else [])
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


def _force_fixed_command(env: Any, fixed_command: tuple[float, float, float] | None) -> None:
    if fixed_command is None:
        return
    try:
        command = env.unwrapped.command_manager.get_term("twist")
    except Exception:
        return
    value = torch.tensor(fixed_command, dtype=torch.float32, device=env.unwrapped.device)
    if hasattr(command, "vel_command_b"):
        command.vel_command_b[:, :] = value
    if hasattr(command, "is_standing_env"):
        command.is_standing_env[:] = False
    if hasattr(command, "is_heading_env"):
        command.is_heading_env[:] = False


def _apply_command_to_full_obs(
    observations: np.ndarray,
    fixed_command: tuple[float, float, float] | None,
) -> np.ndarray:
    if fixed_command is None:
        return observations
    observations = observations.copy()
    observations[:, 9:12] = np.asarray(fixed_command, dtype=np.float32)
    return observations


def _mean(values: list[float], fallback: float = 0.0) -> float:
    return float(np.mean(values)) if values else fallback


def _state_from_full_obs_np(observations: np.ndarray, actuator_force: np.ndarray) -> np.ndarray:
    """Build the 45D RWM state from policy observations plus privileged force."""

    return np.concatenate(
        (observations[:, 0:9], observations[:, 12:36], actuator_force),
        axis=-1,
    ).astype(np.float32, copy=False)


def _actuator_force_np(env: Any) -> np.ndarray:
    return (
        env.scene["robot"]
        .data.actuator_force.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


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
    _configure_fixed_command_range(env_cfg, fixed_command, command_sequence)
    initial_command = command_sequence[0] if command_sequence else fixed_command

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

    dynamics = None
    history_horizon = 0
    if args.model_path:
        dynamics, _ = load_any_go2_dynamics_checkpoint(
            resolve_repo_path(args.model_path),
            device=device,
        )
        dynamics.eval()
        history_horizon = int(dynamics.cfg.history_horizon)

    obs_dict, _ = env.reset()
    _force_fixed_command(env, initial_command)
    full_observations = _apply_command_to_full_obs(_actor_obs(obs_dict), initial_command)

    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    rollout_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    reset_counts = np.zeros(args.num_envs, dtype=np.int64)
    termination_counts = np.zeros(args.num_envs, dtype=np.int64)
    alive_reward_sums = np.zeros(args.num_envs, dtype=np.float64)
    alive_step_counts = np.zeros(args.num_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    terminated_count = 0
    timeout_count = 0
    first_failure_steps = np.full(args.num_envs, -1, dtype=np.int64)
    logged_metrics: dict[str, list[float]] = defaultdict(list)
    base_lin_vel_samples: list[np.ndarray] = []
    base_ang_vel_samples: list[np.ndarray] = []
    command_samples: list[np.ndarray] = []
    action_abs_samples: list[float] = []
    action_delta_abs_samples: list[float] = []
    action_saturation_samples: list[float] = []
    step_vel_error_xy: list[float] = []
    step_vel_error_yaw: list[float] = []
    previous_actions_np: np.ndarray | None = None
    roll_abs_samples: list[float] = []
    pitch_abs_samples: list[float] = []
    tilt_samples: list[float] = []
    epistemic_samples: list[float] = []
    epistemic_valid_rows = 0
    epistemic_total_rows = 0
    estimated_base_lin_vel_samples: list[np.ndarray] = []
    estimated_base_lin_vel_truth_samples: list[np.ndarray] = []
    estimator_confidence_samples: list[np.ndarray] = []
    estimator_per_foot_samples: list[np.ndarray] = []
    estimator_contact_samples: list[np.ndarray] = []

    state_history: torch.Tensor | None = None
    action_history: torch.Tensor | None = None
    history_age: torch.Tensor | None = None
    if dynamics is not None:
        initial_state = torch.from_numpy(
            _state_from_full_obs_np(full_observations, _actuator_force_np(env))
        ).to(device)
        initial_action = torch.from_numpy(full_observations[:, 36:48]).to(device)
        state_history = initial_state[:, None, :].repeat(1, history_horizon, 1)
        action_history = initial_action[:, None, :].repeat(1, history_horizon, 1)
        history_age = torch.zeros(args.num_envs, dtype=torch.long, device=device)

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
                command_sequence[(_step // max(1, args.command_switch_steps)) % len(command_sequence)]
                if command_sequence
                else fixed_command
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
            action_saturation_samples.append(float((np.abs(actions_np) >= 0.95).mean()))
            if previous_actions_np is not None:
                action_delta_abs_samples.append(float(np.abs(actions_np - previous_actions_np).mean()))
            previous_actions_np = actions_np.copy()
            base_lin_vel_samples.append(full_observations[:, 0:3].copy())
            base_ang_vel_samples.append(full_observations[:, 3:6].copy())
            command_samples.append(full_observations[:, 9:12].copy())
            projected_gravity = full_observations[:, 6:9]
            roll_proxy = np.arctan2(projected_gravity[:, 1], -projected_gravity[:, 2])
            pitch_proxy = np.arctan2(
                -projected_gravity[:, 0],
                np.sqrt(np.square(projected_gravity[:, 1]) + np.square(projected_gravity[:, 2])),
            )
            tilt = np.arccos(np.clip(-projected_gravity[:, 2], -1.0, 1.0))
            roll_abs_samples.append(float(np.abs(roll_proxy).mean()))
            pitch_abs_samples.append(float(np.abs(pitch_proxy).mean()))
            tilt_samples.append(float(tilt.mean()))
            step_vel_error_xy.append(
                float(
                    np.linalg.norm(
                        full_observations[:, 0:2] - full_observations[:, 9:11],
                        axis=1,
                    ).mean()
                )
            )
            step_vel_error_yaw.append(
                float(np.abs(full_observations[:, 5] - full_observations[:, 11]).mean())
            )
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
            if dynamics is not None:
                assert state_history is not None and action_history is not None and history_age is not None
                action_history = torch.cat((action_history[:, 1:], actions_t[:, None, :]), dim=1)
                model_ids = torch.zeros(args.num_envs, dtype=torch.long, device=device)
                _, _, epistemic, _, _ = dynamics.predict(state_history, action_history, model_ids)
                valid_history = history_age >= history_horizon - 1
                epistemic_total_rows += int(valid_history.numel())
                if valid_history.any():
                    epistemic_samples.append(float(epistemic[valid_history].mean().cpu()))
                    epistemic_valid_rows += int(valid_history.sum().item())
            obs_dict, rewards, terminateds, truncateds, extras = env.step(actions_t)
            rewards_np = rewards.detach().cpu().numpy().astype(np.float64)
            term_np = terminateds.detach().cpu().numpy().astype(bool)
            trunc_np = truncateds.detach().cpu().numpy().astype(bool)
            done_np = np.logical_or(term_np, trunc_np)

            ep_returns += rewards_np
            rollout_returns += rewards_np
            ep_lengths += 1

            # First-failure statistics describe survival independently of the
            # environment's automatic resets. Include the failure step itself.
            alive_before_step = first_failure_steps < 0
            alive_reward_sums[alive_before_step] += rewards_np[alive_before_step]
            alive_step_counts[alive_before_step] += 1

            first_now = np.logical_and(term_np, first_failure_steps < 0)
            first_failure_steps[first_now] = _step + 1
            reset_counts += done_np.astype(np.int64)
            termination_counts += term_np.astype(np.int64)

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
            if dynamics is not None:
                assert state_history is not None and action_history is not None and history_age is not None
                actual_state = torch.from_numpy(
                    _state_from_full_obs_np(full_observations, _actuator_force_np(env))
                ).to(device)
                state_history = torch.cat((state_history[:, 1:], actual_state[:, None, :]), dim=1)
                history_age += 1
                if done_np.any():
                    done_ids = torch.from_numpy(np.flatnonzero(done_np)).to(device=device, dtype=torch.long)
                    reset_state = actual_state[done_ids]
                    reset_action = torch.from_numpy(full_observations[done_np, 36:48]).to(device)
                    state_history[done_ids] = reset_state[:, None, :].repeat(1, history_horizon, 1)
                    action_history[done_ids] = reset_action[:, None, :].repeat(1, history_horizon, 1)
                    history_age[done_ids] = 0

    env.close()

    # Fixed-horizon policy comparisons must include every reward accrued by
    # every environment, including rewards after auto-reset.  The previous
    # implementation switched to completed episodes when any failure occurred,
    # silently excluding surviving trajectories and producing contradictory
    # termination/episode-length summaries.
    mean_return = float(rollout_returns.mean())
    std_return = float(rollout_returns.std())
    residual_lengths = ep_lengths[ep_lengths > 0].astype(np.float64).tolist()
    all_segment_lengths = [*completed_lengths, *residual_lengths]
    mean_episode_length = _mean(all_segment_lengths, fallback=float(args.steps))
    failed_mask = first_failure_steps >= 0
    first_failure_or_censor = np.where(failed_mask, first_failure_steps, args.steps)
    survival_at = {
        str(horizon): float(
            np.logical_or(~failed_mask, first_failure_steps > horizon).mean()
        )
        for horizon in (50, 100, 200, 500, 1000)
        if horizon <= args.steps
    }
    km_times = [0]
    km_survival = [1.0]
    survival_probability = 1.0
    for event_time in sorted(np.unique(first_failure_steps[failed_mask]).tolist()):
        at_risk = int(np.sum(first_failure_or_censor >= event_time))
        events = int(np.sum(first_failure_steps == event_time))
        if at_risk > 0:
            survival_probability *= 1.0 - events / at_risk
        km_times.append(int(event_time))
        km_survival.append(float(survival_probability))
    base_lin_vel = np.concatenate(base_lin_vel_samples, axis=0) if base_lin_vel_samples else np.zeros((1, 3))
    base_ang_vel = np.concatenate(base_ang_vel_samples, axis=0) if base_ang_vel_samples else np.zeros((1, 3))
    commands = np.concatenate(command_samples, axis=0) if command_samples else np.zeros((1, 3))
    vel_error_xy = np.linalg.norm(base_lin_vel[:, 0:2] - commands[:, 0:2], axis=1)
    yaw_error = np.abs(base_ang_vel[:, 2] - commands[:, 2])

    post_switch_metrics: dict[str, float] = {}
    if command_sequence and args.command_switch_steps > 0:
        switch_starts = list(range(args.command_switch_steps, args.steps, args.command_switch_steps))
        for horizon in (10, 25, 50):
            xy_values = [
                value
                for start in switch_starts
                for value in step_vel_error_xy[start : min(start + horizon, args.steps)]
            ]
            yaw_values = [
                value
                for start in switch_starts
                for value in step_vel_error_yaw[start : min(start + horizon, args.steps)]
            ]
            post_switch_metrics[f"post_switch_error_vel_xy_{horizon}"] = _mean(xy_values)
            post_switch_metrics[f"post_switch_error_vel_yaw_{horizon}"] = _mean(yaw_values)

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
        "mean_return": mean_return,
        "fixed_horizon_return": mean_return,
        "std_return": std_return,
        "mean_episode_length": mean_episode_length,
        "mean_completed_episode_return": _mean(completed_returns),
        "mean_completed_episode_length": _mean(completed_lengths),
        "mean_episode_length_with_right_censoring": mean_episode_length,
        "completed_episodes": len(completed_returns),
        "terminated_count": terminated_count,
        "timeout_count": timeout_count,
        "non_timeout_termination_count": terminated_count,
        "termination_rate": float(failed_mask.mean()),
        "survive_to_1000_ratio": float((~failed_mask).mean()),
        **{f"survival_at_{horizon}": value for horizon, value in survival_at.items()},
        "mean_time_to_first_failure_failed_only": (
            float(first_failure_steps[failed_mask].mean()) if failed_mask.any() else None
        ),
        "restricted_mean_time_to_first_failure": float(first_failure_or_censor.mean()),
        "terminations_per_1000_env_steps": float(
            termination_counts.sum() * 1000.0 / (args.num_envs * args.steps)
        ),
        "mean_resets_per_env": float(reset_counts.mean()),
        "std_resets_per_env": float(reset_counts.std()),
        "reward_per_alive_step": float(
            alive_reward_sums.sum() / max(1, alive_step_counts.sum())
        ),
        "first_failure_steps": first_failure_steps.tolist(),
        "reset_counts_per_env": reset_counts.tolist(),
        "termination_counts_per_env": termination_counts.tolist(),
        "alive_reward_sums_per_env": alive_reward_sums.tolist(),
        "alive_step_counts_per_env": alive_step_counts.tolist(),
        "kaplan_meier": {"time": km_times, "survival": km_survival},
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
        "action_delta_abs_mean": _mean(action_delta_abs_samples),
        "action_saturation_ratio": _mean(action_saturation_samples),
        "roll_abs_mean": _mean(roll_abs_samples),
        "pitch_abs_mean": _mean(pitch_abs_samples),
        "tilt_mean": _mean(tilt_samples),
        "epistemic_uncertainty": _mean(epistemic_samples, fallback=float("nan")),
        "epistemic_valid_fraction": (
            float(epistemic_valid_rows / epistemic_total_rows) if epistemic_total_rows else 0.0
        ),
        **post_switch_metrics,
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
