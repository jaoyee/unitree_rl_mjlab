"""Validate the normal-Go2 offline dataset and write a compact quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


_FINITE_KEYS = (
    "states",
    "actions",
    "raw_actions",
    "next_states",
    "contacts",
    "terminations",
    "observations",
    "next_observations",
    "commands",
    "rewards",
    "prev_actions",
    "noisy_actor_observations",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--minimum_transitions", type=int, default=1_000_000)
    parser.add_argument("--expected_num_envs", type=int, default=1024)
    parser.add_argument(
        "--expected_preset",
        choices=(
            "default",
            "friend_flat",
            "calibrated_default",
            "calibrated_friend_flat",
        ),
        required=True,
    )
    parser.add_argument("--expected_scale", type=float, required=True)
    parser.add_argument("--expected_shards", type=int, default=4)
    return parser.parse_args()


def _tensor_stats(rows: list[torch.Tensor]) -> dict[str, float]:
    count = 0
    total = 0.0
    total_sq = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for row in rows:
        value = row.float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Dataset contains NaN or Inf")
        count += value.numel()
        total += float(value.sum(dtype=torch.float64))
        total_sq += float(value.square().sum(dtype=torch.float64))
        minimum = min(minimum, float(value.min()))
        maximum = max(maximum, float(value.max()))
    mean = total / max(count, 1)
    variance = max(0.0, total_sq / max(count, 1) - mean * mean)
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": variance**0.5,
    }


def validate_dataset(dataset: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if dataset.get("format_version") != "go2_mixed_rwm_dataset_v2":
        raise ValueError(f"Expected v2 dataset, got {dataset.get('format_version')!r}")
    num_envs = int(dataset["num_envs"])
    if num_envs != int(args.expected_num_envs):
        raise ValueError(f"Expected {args.expected_num_envs} envs, got {num_envs}")
    num_steps = len(dataset["states"])
    transitions = num_steps * num_envs
    if transitions < int(args.minimum_transitions):
        raise ValueError(f"Expected at least {args.minimum_transitions} transitions, got {transitions}")

    list_keys = [key for key, value in dataset.items() if isinstance(value, list)]
    for key in list_keys:
        if len(dataset[key]) != num_steps:
            raise ValueError(f"Transition key {key!r} has the wrong number of rows")
        if any(int(row.shape[0]) != num_envs for row in dataset[key]):
            raise ValueError(f"Transition key {key!r} has a malformed vector row")
    finite_stats = {key: _tensor_stats(dataset[key]) for key in _FINITE_KEYS}
    if finite_stats["actions"]["min"] < -1.00001 or finite_stats["actions"]["max"] > 1.00001:
        raise ValueError("Effective actions fall outside [-1, 1]")

    episode_ids = torch.stack(dataset["episode_ids"]).long()
    timesteps = torch.stack(dataset["timesteps"]).long()
    episode_change = episode_ids[1:] != episode_ids[:-1]
    expected_next = timesteps[:-1] + 1
    continuity_error = (~episode_change & (timesteps[1:] != expected_next)).sum().item()
    if continuity_error:
        raise ValueError(f"Found {continuity_error} within-episode timestep discontinuities")
    if bool((timesteps[0] != 0).any()) or bool((timesteps[1:][episode_change] != 0).any()):
        raise ValueError("New episodes do not start at timestep zero")

    startup = dataset.get("startup_domain_table")
    episodes = dataset.get("episode_domain_table")
    if not isinstance(startup, dict) or not isinstance(episodes, dict):
        raise ValueError("Dataset is missing DR provenance tables")
    startup_ids = startup["startup_domain_id"].long()
    table_episode_ids = episodes["episode_id"].long()
    if torch.unique(startup_ids).numel() != startup_ids.numel():
        raise ValueError("startup_domain_id is not unique")
    if torch.unique(table_episode_ids).numel() != table_episode_ids.numel():
        raise ValueError("episode_domain_table episode_id is not unique")
    if not set(torch.unique(episode_ids).tolist()).issubset(set(table_episode_ids.tolist())):
        raise ValueError("Transition episode IDs are missing from episode_domain_table")
    if not set(episodes["startup_domain_id"].tolist()).issubset(set(startup_ids.tolist())):
        raise ValueError("Episode rows reference an unknown startup domain")
    expected_startup_domains = int(args.expected_num_envs) * int(args.expected_shards)
    if int(startup_ids.numel()) != expected_startup_domains:
        raise ValueError(
            f"Expected {expected_startup_domains} startup domains, got {startup_ids.numel()}"
        )

    metadata = dataset.get("metadata") or {}
    env_randomization = metadata.get("env_randomization") or {}
    if str(env_randomization.get("randomization_preset")) != str(args.expected_preset):
        raise ValueError("Dataset randomization preset does not match the branch")
    if abs(float(env_randomization.get("randomization_scale", -1.0)) - float(args.expected_scale)) > 1e-9:
        raise ValueError("Dataset randomization scale does not match the branch")
    if metadata.get("broken_pd_joint_names"):
        raise ValueError("Normal-Go2 dataset unexpectedly contains a broken joint")
    if metadata.get("joint_strength_scales"):
        raise ValueError("Normal-Go2 dataset unexpectedly contains a joint-strength override")
    if int(metadata.get("merged_shards", 0)) != int(args.expected_shards):
        raise ValueError("Dataset does not contain the required independent shards")

    actions = torch.stack(dataset["actions"])
    raw_actions = torch.stack(dataset["raw_actions"])
    action_delta = (actions - raw_actions).abs()
    noisy_actor = torch.stack(dataset["noisy_actor_observations"])
    clean_obs = torch.stack(dataset["observations"])
    sensor_delta = (noisy_actor - clean_obs).abs() if noisy_actor.shape == clean_obs.shape else None
    delay = torch.stack(dataset["actuator_delay_substeps"]).long()
    friction = startup["collision_friction"].float()
    if str(args.expected_preset) in {"friend_flat", "calibrated_friend_flat"}:
        scale = float(args.expected_scale)
        expected_ranges = {
            "kp_scale": (1.0 - 0.1 * scale, 1.0 + 0.1 * scale),
            "kd_scale": (1.0 - 0.1 * scale, 1.0 + 0.1 * scale),
            "motor_strength": (1.0 - 0.2 * scale, 1.0 + 0.2 * scale),
            "motor_zero_offset": (-0.035 * scale, 0.035 * scale),
            "joint_reset_scale": (1.0 - 0.5 * scale, 1.0 + 0.5 * scale),
        }
        for key, (low, high) in expected_ranges.items():
            values = episodes[key].float()
            if float(values.min()) < low - 1e-5 or float(values.max()) > high + 1e-5:
                raise ValueError(f"Episode DR field {key!r} falls outside [{low}, {high}]")
        if str(args.expected_preset) == "calibrated_friend_flat":
            friction_low = 0.8 - 0.2 * scale
            friction_high = 0.8 + 0.2 * scale
        else:
            friction_low = 1.0 - scale
            friction_high = 1.0 + scale
        if float(friction.min()) < friction_low - 1e-5 or float(friction.max()) > friction_high + 1e-5:
            raise ValueError("Friend friction falls outside the configured range")
        friction_spread = friction.max(dim=1).values - friction.min(dim=1).values
        if float(friction_spread.max()) > 1e-5:
            raise ValueError("Friend friction is not shared across collision geoms")
        if int(delay.min()) < 0 or int(delay.max()) > round(4 * scale):
            raise ValueError("Friend actuator delay falls outside the configured range")
    else:
        if int(delay.abs().max()) != 0:
            raise ValueError("Legacy default branch unexpectedly contains friend actuator delay")
        for key in ("kp_scale", "kd_scale", "motor_strength"):
            if not torch.allclose(episodes[key].float(), torch.ones_like(episodes[key].float())):
                raise ValueError(f"Legacy default branch unexpectedly randomizes {key}")
        if not torch.allclose(
            episodes["motor_zero_offset"].float(),
            torch.zeros_like(episodes["motor_zero_offset"].float()),
        ):
            raise ValueError("Legacy default branch unexpectedly applies an effective motor target offset")
        if str(args.expected_preset) == "calibrated_default":
            scale = float(args.expected_scale)
            friction_low = 0.8 - 0.2 * scale
            friction_high = 0.8 + 0.2 * scale
            if (
                float(friction.min()) < friction_low - 1e-5
                or float(friction.max()) > friction_high + 1e-5
            ):
                raise ValueError("Calibrated default friction falls outside the configured range")
            friction_spread = friction.max(dim=1).values - friction.min(dim=1).values
            if float(friction_spread.max()) > 1e-5:
                raise ValueError("Calibrated default friction is not shared across collision geoms")
    delay_hist = torch.bincount(delay.flatten(), minlength=int(delay.max().item()) + 1)

    summary = {
        "status": "completed",
        "format_version": dataset["format_version"],
        "num_envs": num_envs,
        "num_time_steps": num_steps,
        "num_transitions": transitions,
        "startup_domains": int(startup_ids.numel()),
        "episodes": int(table_episode_ids.numel()),
        "transition_episodes": int(torch.unique(episode_ids).numel()),
        "finite_stats": finite_stats,
        "effective_action_clipping_fraction": float((actions.abs() >= 0.999999).float().mean()),
        "raw_to_effective_action_abs_mean": float(action_delta.mean()),
        "clean_to_noisy_actor_abs_mean": None if sensor_delta is None else float(sensor_delta.mean()),
        "actuator_delay_histogram": {str(index): int(value) for index, value in enumerate(delay_hist.tolist())},
        "termination_fraction": float(torch.stack(dataset["dones"]).float().mean()),
        "collector_counts": metadata.get("collector_counts", {}),
        "command_mode_counts": metadata.get("command_mode_counts", {}),
        "env_randomization": env_randomization,
        "source_shards": metadata.get("source_shards", []),
    }
    return summary


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    summary = validate_dataset(dataset, args)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"dataset_gate=completed")
    print(f"summary={output}")
    print(f"num_transitions={summary['num_transitions']}")
    print(f"startup_domains={summary['startup_domains']}")
    print(f"episodes={summary['episodes']}")


if __name__ == "__main__":
    main()
