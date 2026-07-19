#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


MODE_NAMES = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")
MODE_TARGETS = (2000, 6250, 2500, 2000, 3500, 4250, 1250, 3250)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--source-all", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--deploy-yaml", required=True)
    parser.add_argument("--expert-sha256", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stack(data: dict, key: str) -> torch.Tensor:
    value = data[key]
    if isinstance(value, (list, tuple)):
        value = torch.stack([torch.as_tensor(item) for item in value])
    return torch.as_tensor(value)


def tensor_schema(data: dict) -> dict:
    result = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
            value = torch.stack(value)
        if torch.is_tensor(value):
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return result


def command_labels(commands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-3
    nonzero = np.abs(commands) > eps
    mode = np.full(len(commands), -1, dtype=np.int64)
    patterns = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
    for idx, pattern in enumerate(patterns):
        mode[np.all(nonzero == np.asarray(pattern, dtype=bool), axis=1)] = idx
    edges = ((0.05, 0.20, 0.35, 0.50), (0.03, 0.087, 0.143, 0.20), (0.05, 0.167, 0.283, 0.40))
    marginals = np.zeros((len(commands), 18), dtype=np.int64)
    for axis, axis_edges in enumerate(edges):
        values = commands[:, axis]
        magnitude = np.abs(values)
        bins = np.digitize(magnitude, axis_edges[1:-1], right=False)
        valid = (magnitude >= axis_edges[0] - 1e-6) & (magnitude <= axis_edges[-1] + 1e-6)
        for sign_index, sign_mask in enumerate((values < -eps, values > eps)):
            for bin_index in range(3):
                marginals[:, axis * 6 + sign_index * 3 + bin_index] = valid & sign_mask & (bins == bin_index)
    return mode, marginals


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    canonical_path = Path(args.canonical).resolve()
    source_all = Path(args.source_all).resolve()
    target_path = Path(args.target).resolve()
    source_metadata_path = Path(args.source_metadata).resolve()
    deploy_path = Path(args.deploy_yaml).resolve()
    converter_path = Path(args.converter).resolve()
    selector_path = Path(args.selector).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
    source_metadata = json.loads(source_metadata_path.read_text())
    target = json.loads(target_path.read_text())

    metadata = dict(data.get("metadata") or {})
    metadata.update({
        "robot_id": "go2sun",
        "gap": "g0",
        "payload": 0.0,
        "payload_kg": 0.0,
        "rr_calf_strength": 1.0,
        "collection_date": "2026-07-19",
        "expert_policy_reference": source_metadata.get("checkpoint_path"),
        "expert_policy_sha256": args.expert_sha256,
        "deploy_yaml_sha256": sha256(deploy_path),
        "converter_sha256": sha256(converter_path),
        "selector_sha256": sha256(selector_path),
        "base_velocity_estimator_version": "contact_kinematics_affine_ema_v1",
        "command_target_path": str(target_path),
        "command_target_sha256": sha256(target_path),
        "source_dataset_sha256": sha256(source_all),
    })
    data["metadata"] = metadata
    temporary = dataset_path.with_suffix(dataset_path.suffix + ".new")
    torch.save(data, temporary)
    os.replace(temporary, dataset_path)

    commands = stack(data, "commands").reshape(-1, 3).float().numpy()
    actions = stack(data, "actions").reshape(-1, 12).float()
    raw_actions = stack(data, "raw_actions").reshape(-1, 12).float()
    observations = stack(data, "observations").reshape(-1, 45).float()
    next_observations = stack(data, "next_observations").reshape(-1, 45).float()
    states = stack(data, "states").reshape(-1, 45).float()
    next_states = stack(data, "next_states").reshape(-1, 45).float()
    contacts = stack(data, "contacts").reshape(-1, 4).float()
    foot_forces = stack(data, "foot_forces").reshape(-1, 4).float()
    episode_ids = stack(data, "episode_ids").reshape(-1).long()
    timesteps = stack(data, "timesteps").reshape(-1).long()
    confidence = stack(data, "base_lin_vel_confidence").reshape(-1).float()
    terminations = stack(data, "terminations").reshape(-1).float()

    tensors = (actions, raw_actions, observations, next_observations, states, next_states, contacts, foot_forces, confidence)
    all_finite = all(bool(torch.isfinite(value).all()) for value in tensors)
    action_last_error = float((actions - next_observations[:, 33:45]).abs().max())
    links = (episode_ids[1:] == episode_ids[:-1])
    timestep_errors = int(((timesteps[1:] != timesteps[:-1] + 1) & links).sum())
    valid_40 = 0
    block_lengths = []
    start = 0
    for index in range(1, len(episode_ids) + 1):
        contiguous = index < len(episode_ids) and episode_ids[index] == episode_ids[index - 1] and timesteps[index] == timesteps[index - 1] + 1
        if contiguous:
            continue
        length = index - start
        block_lengths.append(length)
        valid_40 += max(0, length - 39)
        start = index

    modes, marginals = command_labels(commands)
    mode_counts = np.bincount(modes[modes >= 0], minlength=8)
    marginal_counts = marginals.sum(axis=0)
    target_counts = np.asarray(target["marginal_target_counts"], dtype=np.int64)
    normalized_l1 = float(np.abs(marginal_counts - target_counts).sum() / max(target_counts.sum(), 1))

    contact_fraction = contacts.mean(0)
    action_abs_mean = actions.abs().mean(0)
    actuator_force = states[:, 33:45].abs().mean(0)
    projected_gravity_norm = observations[:, 3:6].norm(dim=1)
    schema = tensor_schema(data)
    canonical_schema = tensor_schema(canonical)
    schema_common = sorted(set(schema) & set(canonical_schema))
    schema_mismatches = {
        key: {"new": schema[key], "canonical": canonical_schema[key]}
        for key in schema_common if schema[key] != canonical_schema[key]
    }
    missing_from_new = sorted(set(canonical_schema) - set(schema))
    extra_in_new = sorted(set(schema) - set(canonical_schema))

    validation = {
        "status": "pass",
        "transitions": int(len(actions)),
        "format_version": data.get("format_version"),
        "all_finite": all_finite,
        "action_to_next_observation_last_action_max_error": action_last_error,
        "same_episode_timestep_errors": timestep_errors,
        "episode_count": len(block_lengths),
        "continuous_block_count": len(block_lengths),
        "continuous_block_min": min(block_lengths),
        "continuous_block_max": max(block_lengths),
        "valid_40_step_starts": valid_40,
        "measured_dt_mean": metadata.get("measured_dt_mean"),
        "step_dt": metadata.get("step_dt"),
        "base_lin_vel_confidence_mean": float(confidence.mean()),
        "base_lin_vel_confidence_nonzero_fraction": float((confidence > 0).float().mean()),
        "termination_count": int((terminations > 0.5).sum()),
        "contact_order": metadata.get("contact_names"),
        "contact_fraction_per_foot": contact_fraction.tolist(),
        "rr_minus_rl_contact_fraction": float(contact_fraction[2] - contact_fraction[3]),
        "rr_calf_action_abs_mean": float(action_abs_mean[11]),
        "rl_calf_action_abs_mean": float(action_abs_mean[8]),
        "rr_to_rl_calf_action_ratio": float(action_abs_mean[11] / action_abs_mean[8].clamp_min(1e-8)),
        "rr_calf_actuator_force_abs_mean": float(actuator_force[11]),
        "rl_calf_actuator_force_abs_mean": float(actuator_force[8]),
        "rr_to_rl_calf_actuator_force_ratio": float(actuator_force[11] / actuator_force[8].clamp_min(1e-8)),
        "action_saturation_fraction_abs_ge_0p99": float((actions.abs() >= 0.99).float().mean()),
        "projected_gravity_norm_mean": float(projected_gravity_norm.mean()),
        "projected_gravity_norm_std": float(projected_gravity_norm.std()),
        "foot_force_mean_per_foot": foot_forces.mean(0).tolist(),
        "foot_force_quantile_50_per_foot": torch.quantile(foot_forces, 0.5, dim=0).tolist(),
        "foot_force_quantile_90_per_foot": torch.quantile(foot_forces, 0.9, dim=0).tolist(),
        "schema_mismatches": schema_mismatches,
        "missing_tensor_fields_vs_canonical": missing_from_new,
        "extra_tensor_fields_vs_canonical": extra_in_new,
    }
    required = [
        len(actions) == 25000,
        all_finite,
        action_last_error <= 1e-6,
        timestep_errors == 0,
        valid_40 > 0,
        list(mode_counts) == list(MODE_TARGETS),
        metadata.get("contact_names") == ["FR", "FL", "RR", "RL"],
        data.get("format_version") == canonical.get("format_version"),
        not schema_mismatches,
        not missing_from_new,
    ]
    if not all(required):
        validation["status"] = "fail"

    command_distribution = {
        "mode_order": list(MODE_NAMES),
        "mode_counts": dict(zip(MODE_NAMES, map(int, mode_counts), strict=True)),
        "mode_target_counts": dict(zip(MODE_NAMES, MODE_TARGETS, strict=True)),
        "bin_order": ["x_neg_b0", "x_neg_b1", "x_neg_b2", "x_pos_b0", "x_pos_b1", "x_pos_b2", "y_neg_b0", "y_neg_b1", "y_neg_b2", "y_pos_b0", "y_pos_b1", "y_pos_b2", "yaw_neg_b0", "yaw_neg_b1", "yaw_neg_b2", "yaw_pos_b0", "yaw_pos_b1", "yaw_pos_b2"],
        "actual_marginal_counts": marginal_counts.tolist(),
        "target_marginal_counts": target_counts.tolist(),
        "normalized_marginal_l1_error": normalized_l1,
        "stand_fraction": float(mode_counts[0] / len(actions)),
    }
    provenance = {
        "robot_id": "go2sun",
        "gap": "g0",
        "payload_kg": 0.0,
        "rr_calf_strength": 1.0,
        "collection_date": "2026-07-19",
        "expert_policy_reference": source_metadata.get("checkpoint_path"),
        "expert_policy_sha256": args.expert_sha256,
        "deploy_yaml": str(deploy_path),
        "deploy_yaml_sha256": sha256(deploy_path),
        "converter": str(converter_path),
        "converter_sha256": sha256(converter_path),
        "selector": str(selector_path),
        "selector_sha256": sha256(selector_path),
        "base_velocity_estimator_version": "contact_kinematics_affine_ema_v1",
        "source_all": str(source_all),
        "source_all_sha256": sha256(source_all),
        "source_csv_sha256": metadata.get("source_csv_sha256"),
        "target_spec": str(target_path),
        "target_spec_sha256": sha256(target_path),
        "canonical_schema_reference": str(canonical_path),
        "canonical_schema_reference_sha256": sha256(canonical_path),
    }

    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    (output_dir / "command_distribution.json").write_text(json.dumps(command_distribution, indent=2, sort_keys=True) + "\n")
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    card = f"""# go2sun G0 Official 25K Dataset

- Robot: go2sun
- Gap: g0 (payload 0 kg, RR calf strength 1.0)
- Transitions: {len(actions)}
- Continuous blocks: {len(block_lengths)}; valid 40-step starts: {valid_40}
- Policy observation stored: 45 dimensions
- RWM state stored: 45 dimensions
- Current RWM input: full 45-dimensional state plus 12-dimensional action; no state dimensions dropped
- Current RWM supervised output: 45-dimensional next state plus contacts and termination
- Final policy view: 48 dimensions (45 state + 3 command) before dropping base_lin_vel indices 0,1,2; actor input is 45 dimensions
- base_lin_vel supervision: state indices 0,1,2
- Command marginal normalized L1 error against frozen common target: {normalized_l1:.6f}
- Validation status: {validation['status']}
"""
    (output_dir / "dataset_card.md").write_text(card)
    digest_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "sha256.txt":
            digest_lines.append(f"{sha256(path)}  {path.name}")
    (output_dir / "sha256.txt").write_text("\n".join(digest_lines) + "\n")
    print(json.dumps({"validation": validation, "command_distribution": command_distribution}, indent=2))
    if validation["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
