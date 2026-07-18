"""Run an episode-safe sampler and lightweight proprioceptive RWM loss smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import (
    OfflineSamplerConfig,
    OfflineSequenceSampler,
    stack_time_key,
)
from scripts.reinforcement_learning.rwm_dataset.proprioceptive_dynamics import (
    ProprioceptiveDynamicsConfig,
    ProprioceptiveSystemDynamicsEnsemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    cfg = OfflineSamplerConfig(history_horizon=32, forecast_horizon=8, seed=0)
    sampler = OfflineSequenceSampler(dataset, cfg)

    episode_ids = stack_time_key(dataset, "episode_ids").reshape(
        sampler.num_time_steps, sampler.num_envs
    )
    timesteps = stack_time_key(dataset, "timesteps").reshape(
        sampler.num_time_steps, sampler.num_envs
    )
    all_indices = torch.cat((sampler.train_indices, sampler.val_indices), dim=0)
    for start, env_id in all_indices.tolist():
        episode_window = episode_ids[start : start + sampler.seq_len, env_id]
        timestep_window = timesteps[start : start + sampler.seq_len, env_id]
        if not torch.equal(episode_window[1:], episode_window[:-1]):
            raise RuntimeError("sampler accepted a window crossing an episode boundary")
        if not torch.equal(timestep_window[1:], timestep_window[:-1] + 1):
            raise RuntimeError("sampler accepted a window crossing a timestep discontinuity")

    batch = sampler.sample(
        int(args.batch_size),
        device="cpu",
        split="train",
        include_base_lin_vel_confidence=True,
    )
    expected_shapes = (
        (int(args.batch_size), 40, 45),
        (int(args.batch_size), 40, 12),
        (int(args.batch_size), 40, 45),
        (int(args.batch_size), 40, 4),
        (int(args.batch_size), 40, 1),
        (int(args.batch_size), 40),
    )
    actual_shapes = tuple(tuple(value.shape) for value in batch)
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"sample shapes {actual_shapes}, expected {expected_shapes}")

    dynamics_cfg = ProprioceptiveDynamicsConfig(
        input_state_dim=42,
        output_state_dim=45,
        action_dim=12,
        full_state_dim=45,
        full_action_dim=12,
        contact_dim=4,
        termination_dim=1,
        ensemble_size=1,
        history_horizon=32,
        forecast_horizon=8,
        hidden_size=32,
        num_layers=1,
        dropped_state_indices=(0, 1, 2),
        output_dropped_state_indices=(),
        state_loss_ignored_indices=(),
        dropped_action_indices=(),
    )
    dynamics = ProprioceptiveSystemDynamicsEnsemble(dynamics_cfg)
    state_mean, state_std, action_mean, action_std = sampler.stats()
    dynamics.set_normalizers(state_mean, state_std, action_mean, action_std)
    loss = dynamics.compute_loss(*batch, bootstrap=False)
    nonfinite = [name for name, value in loss.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise RuntimeError(f"non-finite loss values: {nonfinite}")

    report = {
        "schema": "go2_v2_sampler_loss_smoke",
        "status": "pass",
        "dataset": str(dataset_path.resolve()),
        "transitions": sampler.num_transitions,
        "train_sequences": int(sampler.train_indices.shape[0]),
        "val_sequences": int(sampler.val_indices.shape[0]),
        "sample_shapes": [list(shape) for shape in actual_shapes],
        "has_base_lin_vel_confidence": sampler.has_base_lin_vel_confidence,
        "losses": {name: float(value.detach()) for name, value in loss.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
