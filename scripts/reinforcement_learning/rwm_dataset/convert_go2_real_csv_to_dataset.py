"""Convert Go2 deployment CSV logs to the offline RWM dataset v2 schema.

Strict conversion expects the deployment logger to record ``policy_obs`` and
``effective_policy_action`` (or ``raw_policy_action``). Legacy logs can be
converted for smoke tests with ``--allow-legacy-reconstruction``, but those
datasets are explicitly marked approximate in metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STATE_DIM = 45
ACTION_DIM = 12
CONTACT_DIM = 4
OBS_DIM = 45
BASE_LIN_VEL_INDICES = (0, 1, 2)
SDK_JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
CONTACT_NAMES = ("FR", "FL", "RR", "RL")


@dataclass(frozen=True)
class DeploySpec:
    joint_ids_map: tuple[int, ...]
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    action_offset: tuple[float, ...]
    command_ranges: tuple[tuple[float, float], ...]
    step_dt: float


@dataclass
class LoggedRow:
    source: str
    source_row: int
    time: float
    policy_name: str | None
    policy_state_id: int | None
    episode_step: int | None
    unsafe: bool
    state: torch.Tensor
    action: torch.Tensor
    raw_action: torch.Tensor
    observation: torch.Tensor
    command: torch.Tensor
    contact: torch.Tensor
    action_source: str
    exact_policy_observation: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--csv", dest="csv_paths", action="append", required=True)
    parser.add_argument("--deploy-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--max-gap-seconds", type=float, default=0.06)
    parser.add_argument("--max-action-alignment-error", type=float, default=1e-3)
    parser.add_argument("--contact-force-threshold", type=float, default=10.0)
    parser.add_argument(
        "--allow-legacy-reconstruction",
        action="store_true",
        help="Allow missing policy_obs/action fields to be reconstructed approximately.",
    )
    parser.add_argument(
        "--action-source",
        choices=("auto", "effective_policy_action", "raw_policy_action", "q_des_inverse"),
        default="auto",
    )
    parser.add_argument("--minimum-transitions", type=int, default=40)
    return parser.parse_args()


def _as_float_list(value: Any, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values, got {value!r}")
    return tuple(float(item) for item in value)


def _load_deploy_spec(path: Path) -> DeploySpec:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    joint_ids_map = tuple(int(item) for item in cfg["joint_ids_map"])
    if len(joint_ids_map) != ACTION_DIM or sorted(joint_ids_map) != list(range(ACTION_DIM)):
        raise ValueError("deploy.yaml joint_ids_map must be a permutation of 0..11")
    actions = cfg["actions"]["JointPositionAction"]
    default_joint_pos = _as_float_list(cfg["default_joint_pos"], "default_joint_pos", ACTION_DIM)
    action_scale = _as_float_list(actions["scale"], "actions.JointPositionAction.scale", ACTION_DIM)
    action_offset = _as_float_list(actions["offset"], "actions.JointPositionAction.offset", ACTION_DIM)
    ranges = cfg["commands"]["base_velocity"]["ranges"]
    command_ranges = tuple(
        tuple(float(v) for v in ranges[key])
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    )
    return DeploySpec(
        joint_ids_map=joint_ids_map,
        joint_names=tuple(SDK_JOINT_NAMES[idx] for idx in joint_ids_map),
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        action_offset=action_offset,
        command_ranges=command_ranges,
        step_dt=float(cfg.get("step_dt", 0.02)),
    )


def _has_vector(fieldnames: set[str], prefix: str, size: int) -> bool:
    return all(f"{prefix}_{idx}" in fieldnames for idx in range(size))


def _vector(row: dict[str, str], prefix: str, size: int) -> list[float]:
    values = []
    for idx in range(size):
        raw = row.get(f"{prefix}_{idx}", "")
        if raw is None or not raw.strip():
            raise ValueError(f"missing {prefix}_{idx}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"non-finite {prefix}_{idx}")
        values.append(value)
    return values


def _optional_float(row: dict[str, str], name: str) -> float | None:
    raw = row.get(name)
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _scale_joystick(value: float, limits: tuple[float, float]) -> float:
    return value * (limits[1] if value > 0.0 else -limits[0])


def _projected_gravity_from_rpy(rpy: Sequence[float]) -> list[float]:
    roll, pitch, _yaw = rpy
    return [
        math.sin(pitch),
        -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    ]


def _policy_obs_parts(policy_obs: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
    if len(policy_obs) == 45:
        return list(policy_obs[3:6]), list(policy_obs[6:9]), list(policy_obs[33:45])
    if len(policy_obs) == 48:
        return list(policy_obs[6:9]), list(policy_obs[9:12]), list(policy_obs[36:48])
    raise ValueError(f"policy_obs must have 45 or 48 values, got {len(policy_obs)}")


def _proprioceptive_obs(policy_obs: Sequence[float]) -> torch.Tensor:
    if len(policy_obs) == 45:
        return torch.tensor(policy_obs, dtype=torch.float32)
    if len(policy_obs) == 48:
        return torch.tensor(policy_obs[3:], dtype=torch.float32)
    raise ValueError(f"policy_obs must have 45 or 48 values, got {len(policy_obs)}")


def _select_action(
    row: dict[str, str],
    fieldnames: set[str],
    spec: DeploySpec,
    requested: str,
    allow_legacy: bool,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    available = {
        name: _has_vector(fieldnames, name, ACTION_DIM)
        for name in ("effective_policy_action", "raw_policy_action", "q_des")
    }
    source = requested
    if source == "auto":
        if available["effective_policy_action"]:
            source = "effective_policy_action"
        elif available["raw_policy_action"]:
            source = "raw_policy_action"
        elif allow_legacy and available["q_des"]:
            source = "q_des_inverse"
        else:
            raise ValueError(
                "CSV lacks effective_policy_action/raw_policy_action; rebuild the deployment logger "
                "or pass --allow-legacy-reconstruction for a smoke-test-only conversion"
            )
    prefix = "q_des" if source == "q_des_inverse" else source
    if not available.get(prefix, False):
        raise ValueError(f"requested action source {source!r} is not present in the CSV")
    values = _vector(row, prefix, ACTION_DIM)
    if source == "q_des_inverse":
        if not allow_legacy:
            raise ValueError("q_des_inverse requires --allow-legacy-reconstruction")
        values = [
            (value - offset) / scale
            for value, offset, scale in zip(values, spec.action_offset, spec.action_scale, strict=True)
        ]
    action = torch.tensor(values, dtype=torch.float32)
    raw_values = (
        _vector(row, "raw_policy_action", ACTION_DIM)
        if available["raw_policy_action"]
        else values
    )
    return action, torch.tensor(raw_values, dtype=torch.float32), source


def _parse_logged_row(
    row: dict[str, str],
    *,
    source: Path,
    source_row: int,
    fieldnames: set[str],
    spec: DeploySpec,
    args: argparse.Namespace,
) -> LoggedRow:
    time_value = _optional_float(row, "time")
    if time_value is None:
        raise ValueError("missing time")
    action, raw_action, action_source = _select_action(
        row, fieldnames, spec, args.action_source, args.allow_legacy_reconstruction
    )
    if float(action.abs().max()) > 1.0001:
        raise ValueError(
            f"normalized action is outside [-1, 1] (abs max={float(action.abs().max()):.6f})"
        )
    q_sdk = _vector(row, "q", ACTION_DIM)
    dq_sdk = _vector(row, "dq", ACTION_DIM)
    tau_sdk = _vector(row, "tau", ACTION_DIM)
    q = [q_sdk[idx] for idx in spec.joint_ids_map]
    dq = [dq_sdk[idx] for idx in spec.joint_ids_map]
    tau = [tau_sdk[idx] for idx in spec.joint_ids_map]
    q_rel = [value - default for value, default in zip(q, spec.default_joint_pos, strict=True)]
    ang_vel = _vector(row, "ang_vel", 3)

    policy_obs_size = 48 if _has_vector(fieldnames, "policy_obs", 48) else 45
    has_policy_obs = _has_vector(fieldnames, "policy_obs", policy_obs_size)
    if has_policy_obs:
        policy_obs = _vector(row, "policy_obs", policy_obs_size)
        gravity, command, _last_action = _policy_obs_parts(policy_obs)
        observation = _proprioceptive_obs(policy_obs)
    else:
        if not args.allow_legacy_reconstruction:
            raise ValueError(
                "CSV lacks policy_obs; rebuild the deployment logger or pass "
                "--allow-legacy-reconstruction for a smoke-test-only conversion"
            )
        gravity = _projected_gravity_from_rpy(_vector(row, "imu_rpy", 3))
        fixed_command = _vector(row, "cmd_fixed", 3)
        if any(abs(value) > 1e-8 for value in fixed_command):
            command = fixed_command
        else:
            joystick = _vector(row, "cmd_ns", 3)
            command = [
                _scale_joystick(value, limits)
                for value, limits in zip(joystick, spec.command_ranges, strict=True)
            ]
        observation = torch.empty(OBS_DIM, dtype=torch.float32)

    state = torch.tensor(
        [0.0, 0.0, 0.0] + ang_vel + gravity + q_rel + dq + tau,
        dtype=torch.float32,
    )
    if not has_policy_obs:
        # The last-action block is patched after episode segmentation.
        observation[:] = torch.tensor(ang_vel + gravity + command + q_rel + dq + [0.0] * 12)

    if _has_vector(fieldnames, "foot_contact", CONTACT_DIM):
        contact = _vector(row, "foot_contact", CONTACT_DIM)
    else:
        force = _vector(row, "foot_force", CONTACT_DIM)
        contact = [float(value > args.contact_force_threshold) for value in force]

    policy_state = _optional_float(row, "policy_state_id")
    episode_step = _optional_float(row, "episode_step")
    unsafe = bool((_optional_float(row, "unsafe_orientation") or 0.0) > 0.5)
    policy_name = (row.get("policy_name") or "").strip() or None
    return LoggedRow(
        source=str(source),
        source_row=source_row,
        time=time_value,
        policy_name=policy_name,
        policy_state_id=None if policy_state is None else int(policy_state),
        episode_step=None if episode_step is None else int(episode_step),
        unsafe=unsafe,
        state=state,
        action=action,
        raw_action=raw_action,
        observation=observation,
        command=torch.tensor(command, dtype=torch.float32),
        contact=torch.tensor(contact, dtype=torch.float32),
        action_source=action_source,
        exact_policy_observation=has_policy_obs,
    )


def _read_csv(path: Path, spec: DeploySpec, args: argparse.Namespace) -> tuple[list[LoggedRow], int]:
    rows: list[LoggedRow] = []
    rejected = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = set(reader.fieldnames)
        has_policy_obs = _has_vector(fieldnames, "policy_obs", 45)
        has_exact_action = any(
            _has_vector(fieldnames, prefix, ACTION_DIM)
            for prefix in ("effective_policy_action", "raw_policy_action")
        )
        if not args.allow_legacy_reconstruction and not has_policy_obs:
            raise ValueError(
                f"{path} lacks policy_obs_0..44; rebuild/redeploy the updated Go2 logger"
            )
        if not args.allow_legacy_reconstruction and not has_exact_action:
            raise ValueError(
                f"{path} lacks effective_policy_action_0..11/raw_policy_action_0..11; "
                "rebuild/redeploy the updated Go2 logger"
            )
        for source_row, row in enumerate(reader, start=2):
            try:
                rows.append(
                    _parse_logged_row(
                        row,
                        source=path,
                        source_row=source_row,
                        fieldnames=fieldnames,
                        spec=spec,
                        args=args,
                    )
                )
            except (KeyError, TypeError, ValueError):
                rejected += 1
    if len(rows) < 2:
        raise ValueError(f"{path} has fewer than two usable rows")
    return rows, rejected


def _is_boundary(current: LoggedRow, following: LoggedRow, max_gap: float) -> bool:
    if current.source != following.source:
        return True
    dt = following.time - current.time
    if dt <= 0.0 or dt > max_gap:
        return True
    if current.unsafe:
        return True
    if current.policy_name != following.policy_name:
        return True
    if current.policy_state_id != following.policy_state_id:
        return True
    if (
        current.episode_step is not None
        and following.episode_step is not None
        and following.episode_step <= current.episode_step
    ):
        return True
    return False


def _append(dataset: dict[str, Any], key: str, value: torch.Tensor) -> None:
    dataset[key].append(value.unsqueeze(0))


def _build_dataset(rows: list[LoggedRow], spec: DeploySpec, args: argparse.Namespace) -> dict[str, Any]:
    list_keys = (
        "states", "actions", "raw_actions", "env_action_delay_steps",
        "actuator_delay_substeps", "next_states", "contacts", "terminations",
        "observations", "next_observations", "commands", "rewards", "dones",
        "timeouts", "prev_actions", "episode_ids", "timesteps", "collector_types",
        "noisy_actor_observations",
    )
    dataset: dict[str, Any] = {key: [] for key in list_keys}
    dataset.update(
        {
            "format_version": "go2_mixed_rwm_dataset_v2",
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "contact_dim": CONTACT_DIM,
            "termination_dim": 1,
            "num_envs": 1,
            "capacity": 0,
        }
    )

    episode_id = 0
    timestep = 0
    approximate_rows = 0
    dt_values: list[float] = []
    action_alignment_errors: list[float] = []
    episode_ids_seen: set[int] = set()
    previous_action = torch.zeros(ACTION_DIM)

    for idx in range(len(rows) - 1):
        current = rows[idx]
        following = rows[idx + 1]
        if _is_boundary(current, following, args.max_gap_seconds):
            episode_id += 1
            timestep = 0
            previous_action = torch.zeros(ACTION_DIM)
            continue

        if not current.exact_policy_observation:
            current.observation[-ACTION_DIM:] = previous_action
        if not following.exact_policy_observation:
            following.observation[-ACTION_DIM:] = current.action
        else:
            action_alignment_errors.append(
                float((following.observation[-ACTION_DIM:] - current.action).abs().max())
            )

        termination = torch.tensor([float(following.unsafe)], dtype=torch.float32)
        _append(dataset, "states", current.state)
        _append(dataset, "actions", current.action)
        _append(dataset, "raw_actions", current.raw_action)
        _append(dataset, "next_states", following.state)
        _append(dataset, "contacts", following.contact)
        _append(dataset, "terminations", termination)
        _append(dataset, "observations", current.observation)
        _append(dataset, "next_observations", following.observation)
        _append(dataset, "commands", current.command)
        _append(dataset, "prev_actions", current.observation[-ACTION_DIM:])
        _append(dataset, "noisy_actor_observations", current.observation)
        _append(dataset, "env_action_delay_steps", torch.tensor(0, dtype=torch.long))
        _append(dataset, "actuator_delay_substeps", torch.tensor(0, dtype=torch.long))
        _append(dataset, "rewards", torch.tensor(0.0))
        _append(dataset, "dones", termination.bool().squeeze(0))
        _append(dataset, "timeouts", torch.tensor(False))
        _append(dataset, "episode_ids", torch.tensor(episode_id, dtype=torch.long))
        _append(dataset, "timesteps", torch.tensor(timestep, dtype=torch.long))
        _append(dataset, "collector_types", torch.tensor(1, dtype=torch.long))

        approximate_rows += int(not current.exact_policy_observation or current.action_source == "q_des_inverse")
        episode_ids_seen.add(episode_id)
        dt_values.append(following.time - current.time)
        previous_action = current.action
        timestep += 1

    transitions = len(dataset["states"])
    if transitions < args.minimum_transitions:
        raise ValueError(f"only {transitions} valid transitions; require {args.minimum_transitions}")
    dataset["capacity"] = transitions
    dataset["metadata"] = {
        "obs_dim": OBS_DIM,
        "source": "real_go2_csv",
        "conversion_kind": "exact" if approximate_rows == 0 else "approximate_legacy",
        "source_csvs": sorted({row.source for row in rows}),
        "num_input_rows": len(rows),
        "num_time_steps": transitions,
        "num_transitions": transitions,
        "num_episodes": len(episode_ids_seen),
        "state_layout": [
            "base_lin_vel_b[3]", "base_ang_vel_b[3]", "projected_gravity_b[3]",
            "joint_pos_rel[12]", "joint_vel[12]", "actuator_force[12]",
        ],
        "state_loss_ignored_indices": list(BASE_LIN_VEL_INDICES),
        "base_lin_vel_source": "zero_placeholder_unsupervised",
        "policy_observation_kind": "proprioceptive_45d",
        "policy_observation_layout": [
            "base_ang_vel_b[3]", "projected_gravity_b[3]", "command[3]",
            "joint_pos_rel[12]", "joint_vel[12]", "last_action[12]",
        ],
        "joint_ids_map": list(spec.joint_ids_map),
        "joint_names": list(spec.joint_names),
        "sdk_joint_names": list(SDK_JOINT_NAMES),
        "contact_names": list(CONTACT_NAMES),
        "step_dt": spec.step_dt,
        "measured_dt_mean": sum(dt_values) / len(dt_values),
        "measured_dt_min": min(dt_values),
        "measured_dt_max": max(dt_values),
        "action_sources": sorted({row.action_source for row in rows}),
        "max_next_obs_action_alignment_error": max(action_alignment_errors, default=None),
        "reward_available": False,
        "collector_counts": {"expert": transitions},
    }
    alignment_error = dataset["metadata"]["max_next_obs_action_alignment_error"]
    if approximate_rows == 0 and alignment_error is not None:
        if alignment_error > float(args.max_action_alignment_error):
            raise ValueError(
                "strict transition alignment failed: next policy_obs last_action differs from "
                f"the preceding action by {alignment_error:.6g}, limit is "
                f"{args.max_action_alignment_error:.6g}"
            )
    return dataset


def _validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    transitions = len(dataset["states"])
    for key, value in dataset.items():
        if isinstance(value, list):
            if len(value) != transitions:
                raise ValueError(f"{key} has {len(value)} rows, expected {transitions}")
            if not all(torch.isfinite(item.float()).all() for item in value):
                raise ValueError(f"{key} contains NaN or Inf")
    expected_shapes = {
        "states": (1, 45), "actions": (1, 12), "next_states": (1, 45),
        "contacts": (1, 4), "terminations": (1, 1),
        "observations": (1, 45), "next_observations": (1, 45),
    }
    for key, shape in expected_shapes.items():
        if tuple(dataset[key][0].shape) != shape:
            raise ValueError(f"{key} shape is {tuple(dataset[key][0].shape)}, expected {shape}")
    actions = torch.stack(dataset["actions"])
    if float(actions.abs().max()) > 1.0001:
        raise ValueError(f"actions exceed [-1, 1]: abs max={float(actions.abs().max()):.6f}")
    from scripts.reinforcement_learning.rwm_dataset.dataset import (
        OfflineSamplerConfig,
        OfflineSequenceSampler,
    )

    sampler = OfflineSequenceSampler(
        dataset,
        OfflineSamplerConfig(history_horizon=32, forecast_horizon=8, train_fraction=0.95, seed=0),
    )
    sample = sampler.sample(batch_size=2, device="cpu")
    return {
        "transitions": transitions,
        "sampler": sampler.metadata(),
        "sample_shapes": [list(tensor.shape) for tensor in sample],
        "action_abs_max": float(actions.abs().max()),
    }


def main() -> None:
    args = _parse_args()
    deploy_yaml = Path(args.deploy_yaml).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    spec = _load_deploy_spec(deploy_yaml)
    all_rows: list[LoggedRow] = []
    rejected_rows = 0
    for value in args.csv_paths:
        rows, rejected = _read_csv(Path(value).expanduser().resolve(), spec, args)
        all_rows.extend(rows)
        rejected_rows += rejected
    dataset = _build_dataset(all_rows, spec, args)
    validation = _validate_dataset(dataset)
    dataset["metadata"]["rejected_input_rows"] = rejected_rows
    dataset["metadata"]["deploy_yaml"] = str(deploy_yaml)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output)

    report = {
        "output": str(output),
        "format_version": dataset["format_version"],
        "conversion_kind": dataset["metadata"]["conversion_kind"],
        "rejected_input_rows": rejected_rows,
        **validation,
        "metadata": dataset["metadata"],
    }
    report_path = Path(args.report_json).expanduser().resolve() if args.report_json else output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
