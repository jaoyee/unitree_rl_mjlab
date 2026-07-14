"""Focused invariants for the normal-Go2 friend DR conversion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.envs.mjlab import configure_mjlab_randomization, mjlab_randomization_manifest
from flash_rl.envs.mjlab_dr import FriendDelayedActuator, friend_dr_runtime_snapshot
from scripts.reinforcement_learning.rwm_flashsac.world_model_env import (
    interface_observation_noise_half_range,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--scale", type=float, choices=(0.5, 1.0), default=1.0)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _pure_checks() -> dict[str, object]:
    generator = torch.Generator().manual_seed(17)
    raw_torque = torch.randn(4096, 12, generator=generator) * 80.0
    limits = torch.linspace(20.0, 40.0, 12)
    strength = torch.empty(4096, 12).uniform_(0.8, 1.2, generator=generator)
    source = strength * torch.clamp(raw_torque, -limits, limits)
    mapped = torch.clamp(strength * raw_torque, -strength * limits, strength * limits)
    parity_error = float((source - mapped).abs().max())
    if parity_error > 1e-5:
        raise AssertionError(f"Motor-strength saturation mapping error: {parity_error}")

    deployment = interface_observation_noise_half_range("deployment_small", device="cpu") * 0.25
    expected = torch.zeros(48)
    expected[3:6] = 0.0125
    expected[6:9] = 0.0125
    expected[12:24] = 0.0025
    expected[24:36] = 0.01875
    torch.testing.assert_close(deployment, expected)

    deploy_cfg = OmegaConf.load(
        "deploy/robots/go2/config/policy/velocity/rwm_proprioceptive_normal/params/deploy.yaml"
    )
    assert list(deploy_cfg.stiffness) == [30, 40, 50] * 4
    assert list(deploy_cfg.damping) == [1.5, 2, 2.5] * 4
    assert list(deploy_cfg.default_joint_pos) == [0.0, 0.8, -1.5] * 4
    assert list(deploy_cfg.commands.base_velocity.ranges.lin_vel_x) == [-0.5, 0.5]
    assert list(deploy_cfg.commands.base_velocity.ranges.lin_vel_y) == [-0.25, 0.25]
    assert list(deploy_cfg.commands.base_velocity.ranges.ang_vel_z) == [-0.5, 0.5]
    return {
        "motor_strength_parity_max_abs_error": parity_error,
        "deployment_small_group_half_ranges_at_scale_0p25": {
            "base_ang_vel": float(deployment[3]),
            "projected_gravity": float(deployment[6]),
            "joint_pos": float(deployment[12]),
            "joint_vel": float(deployment[24]),
        },
    }


def _environment_checks(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert")
    cfg.scene.num_envs = int(args.num_envs)
    cfg.seed = int(args.seed)
    robot_cfg = cfg.scene.entities["robot"]
    assert robot_cfg.init_state is not None
    assert dict(robot_cfg.init_state.joint_pos) == {
        ".*hip_joint": 0.0,
        ".*thigh_joint": 0.8,
        ".*calf_joint": -1.5,
    }
    twist_cfg = cfg.commands["twist"]
    assert tuple(twist_cfg.ranges.lin_vel_x) == (-0.8, 0.8)
    assert tuple(twist_cfg.ranges.lin_vel_y) == (-0.3, 0.3)
    assert tuple(twist_cfg.ranges.ang_vel_z) == (-0.6, 0.6)
    assert "command_vel" not in cfg.curriculum
    assert float(cfg.rewards["base_height_l2"].params["target_height"]) == 0.32
    configure_mjlab_randomization(
        cfg,
        use_domain_randomization=True,
        use_push_randomization=True,
        use_observation_noise=True,
        randomization_preset="calibrated_friend_flat",
        randomization_components="friction,mass_com,motor,delay,observation,push,initial_state",
        randomization_scale=float(args.scale),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        obs, _ = env.reset()
        assert tuple(obs["actor"].shape) == (args.num_envs, 45)
        assert tuple(obs["critic"].shape) == (args.num_envs, 48)
        assert abs(float(env.cfg.sim.mujoco.timestep) - 0.005) < 1e-12
        assert int(env.cfg.decimation) == 4

        robot = env.scene["robot"]
        joint_names = list(robot.joint_names)
        default_joint_pos = robot.data.default_joint_pos[0]
        expected_default_joint_pos = torch.tensor(
            [
                0.0 if "hip_joint" in name else 0.8 if "thigh_joint" in name else -1.5
                for name in joint_names
            ],
            device=env.device,
        )
        torch.testing.assert_close(default_joint_pos, expected_default_joint_pos)
        assert all(isinstance(actuator, FriendDelayedActuator) for actuator in robot.actuators)
        shared_states = {id(actuator._friend_shared_delay_state) for actuator in robot.actuators}
        assert len(shared_states) == 1

        actions = torch.zeros(args.num_envs, 12, device=env.device)
        _, rewards, _, _, _ = env.step(actions)
        assert bool(torch.isfinite(rewards).all())
        runtime = friend_dr_runtime_snapshot(env)
        max_delay = round(4 * args.scale)
        delay = runtime["actuator_delay_substeps"]
        assert int(delay.min()) >= 0 and int(delay.max()) <= max_delay

        collision_local = [
            index for index, name in enumerate(robot.geom_names) if name.endswith("_collision")
        ]
        collision_global = robot.indexing.geom_ids[
            torch.tensor(collision_local, device=env.device)
        ].long()
        friction = env.sim.model.geom_friction[:, collision_global, 0]
        friction_lo = 0.8 - 0.2 * args.scale
        friction_hi = 0.8 + 0.2 * args.scale
        assert float(friction.min()) >= friction_lo - 1e-6
        assert float(friction.max()) <= friction_hi + 1e-6
        assert float((friction.max(dim=1).values - friction.min(dim=1).values).max()) < 1e-6

        base_id = int(robot.indexing.body_ids[0])
        default_mass = env.sim.get_default_field("body_mass")[base_id]
        default_inertia = env.sim.get_default_field("body_inertia")[base_id]
        mass_scale = env.sim.model.body_mass[:, base_id] / default_mass
        inertia_scale = env.sim.model.body_inertia[:, base_id] / default_inertia
        torch.testing.assert_close(inertia_scale, mass_scale[:, None].expand_as(inertia_scale))

        default_gain = env.sim.get_default_field("actuator_gainprm")
        default_bias = env.sim.get_default_field("actuator_biasprm")
        default_force = env.sim.get_default_field("actuator_forcerange")
        # MuJoCo actuator order is not guaranteed to match entity joint order.
        expected_kp = torch.tensor(
            [30.0] * 4 + [40.0] * 4 + [50.0] * 4,
            device=env.device,
        )
        torch.testing.assert_close(
            torch.sort(default_gain[:, 0]).values,
            torch.sort(expected_kp).values,
        )
        kp = runtime["kp_scale"]
        kd = runtime["kd_scale"]
        motor = runtime["motor_strength"]
        offset = runtime["motor_zero_offset"]
        expected_gain = default_gain[:, 0] * kp * motor
        torch.testing.assert_close(env.sim.model.actuator_gainprm[:, :, 0], expected_gain)
        torch.testing.assert_close(
            env.sim.model.actuator_biasprm[:, :, 0],
            default_bias[:, 0] * motor + expected_gain * offset,
        )
        torch.testing.assert_close(
            env.sim.model.actuator_biasprm[:, :, 1],
            default_bias[:, 1] * kp * motor,
        )
        torch.testing.assert_close(
            env.sim.model.actuator_biasprm[:, :, 2],
            default_bias[:, 2] * kd * motor,
        )
        torch.testing.assert_close(
            env.sim.model.actuator_forcerange[:, :, :],
            default_force.unsqueeze(0) * motor.unsqueeze(-1),
        )

        scale_lo = 1.0 - 0.5 * args.scale
        scale_hi = 1.0 + 0.5 * args.scale
        joint_reset_scale = runtime["joint_reset_scale"]
        assert float(joint_reset_scale.min()) >= scale_lo - 1e-6
        assert float(joint_reset_scale.max()) <= scale_hi + 1e-6

        return {
            "actor_shape": list(obs["actor"].shape),
            "critic_shape": list(obs["critic"].shape),
            "friction_min": float(friction.min()),
            "friction_max": float(friction.max()),
            "base_mass_scale_min": float(mass_scale.min()),
            "base_mass_scale_max": float(mass_scale.max()),
            "delay_values": sorted(set(delay.tolist())),
            "joint_names": joint_names,
            "default_joint_pos": default_joint_pos.tolist(),
            "default_kp": default_gain[:, 0].tolist(),
            "runtime_shapes": {key: list(value.shape) for key, value in runtime.items()},
            "manifest": mjlab_randomization_manifest(
                "calibrated_friend_flat", None, args.scale
            ),
        }
    finally:
        env.close()


def _clean_stance_checks(args: argparse.Namespace) -> dict[str, object]:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert")
    cfg.scene.num_envs = int(args.num_envs)
    cfg.seed = int(args.seed) + 1
    configure_mjlab_randomization(
        cfg,
        use_domain_randomization=False,
        use_push_randomization=False,
        use_observation_noise=False,
        randomization_preset="calibrated_default",
        randomization_scale=0.0,
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        env.reset()
        robot = env.scene["robot"]
        action = torch.zeros(args.num_envs, 12, device=env.device)
        termination_count = 0
        for _ in range(100):
            command = env.command_manager.get_term("twist")
            command.vel_command_b.zero_()
            if hasattr(command, "is_standing_env"):
                command.is_standing_env[:] = True
            _, rewards, terminated, truncated, _ = env.step(action)
            assert bool(torch.isfinite(rewards).all())
            termination_count += int((terminated | truncated).sum())
        height = robot.data.root_link_pos_w[:, 2]
        joint_error = robot.data.joint_pos - robot.data.default_joint_pos
        mean_height = float(height.mean())
        mean_joint_error = float(joint_error.abs().mean())
        if termination_count:
            raise AssertionError(
                f"Clean FixStand terminated {termination_count} times in 100 steps"
            )
        if not 0.30 <= mean_height <= 0.42:
            raise AssertionError(f"Unexpected clean FixStand mean height: {mean_height}")
        if mean_joint_error > 0.10:
            raise AssertionError(
                f"Clean FixStand joint tracking error is too large: {mean_joint_error}"
            )
        return {
            "steps": 100,
            "termination_count": termination_count,
            "mean_base_height": mean_height,
            "mean_abs_joint_error": mean_joint_error,
        }
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    summary = {
        "status": "completed",
        "scale": float(args.scale),
        "pure": _pure_checks(),
        "environment": _environment_checks(args),
        "clean_stance": _clean_stance_checks(args),
    }
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"summary={output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
