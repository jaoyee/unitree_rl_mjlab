#!/usr/bin/env python3
"""Compute the three fixed-horizon metrics used for TRACE comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json, sha256_path


def compute_three_metrics(
    terminated: np.ndarray,
    base_linear_velocity: np.ndarray,
    base_yaw_velocity: np.ndarray,
    command: np.ndarray,
) -> dict[str, Any]:
    terminated = np.asarray(terminated, dtype=bool)
    base_linear_velocity = np.asarray(base_linear_velocity, dtype=np.float64)
    base_yaw_velocity = np.asarray(base_yaw_velocity, dtype=np.float64)
    command = np.asarray(command, dtype=np.float64)
    if terminated.ndim != 2:
        raise ValueError("terminated must have shape [steps, environments].")
    steps, environments = terminated.shape
    if steps < 1 or environments < 1:
        raise ValueError("Evaluation arrays must be non-empty.")
    if base_linear_velocity.shape not in {
        (steps, environments, 2),
        (steps, environments, 3),
    }:
        raise ValueError(
            "base_linear_velocity must have shape [steps, environments, 2 or 3]."
        )
    if base_yaw_velocity.shape != (steps, environments):
        raise ValueError("base_yaw_velocity must have shape [steps, environments].")
    if command.shape != (steps, environments, 3):
        raise ValueError("command must have shape [steps, environments, 3].")
    if (
        not np.isfinite(base_linear_velocity).all()
        or not np.isfinite(base_yaw_velocity).all()
        or not np.isfinite(command).all()
    ):
        raise ValueError("Evaluation velocity or command contains non-finite values.")
    first_failure = np.full(environments, steps, dtype=np.int64)
    for environment in range(environments):
        indices = np.flatnonzero(terminated[:, environment])
        if len(indices):
            first_failure[environment] = int(indices[0])
    survival_steps = np.minimum(first_failure + 1, steps)
    success = first_failure == steps
    failed_before = np.concatenate(
        [
            np.zeros((1, environments), dtype=bool),
            np.maximum.accumulate(terminated[:-1], axis=0),
        ],
        axis=0,
    )
    valid = ~failed_before
    linear_error = np.linalg.norm(
        base_linear_velocity[..., :2] - command[..., :2],
        axis=-1,
    )
    yaw_error = np.abs(base_yaw_velocity - command[..., 2])
    return {
        "schema": "portable_trace_three_metrics_v1",
        "horizon_steps": int(steps),
        "environment_count": int(environments),
        "success_percent": float(100.0 * success.mean()),
        "success_count": int(success.sum()),
        "survival_steps_mean": float(survival_steps.mean()),
        "linear_velocity_error_mps": float(linear_error[valid].mean()),
        "yaw_velocity_error_radps": float(yaw_error[valid].mean()),
        "terminated_environment_count": int((~success).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--rollout-npz", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.rollout_npz).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        report = compute_three_metrics(
            payload["terminated"],
            payload["base_linear_velocity"],
            payload["base_yaw_velocity"],
            payload["command"],
        )
    report["rollout_path"] = str(source)
    report["rollout_sha256"] = sha256_path(source)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
