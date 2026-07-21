"""Go2 trajectory features and Bradley-Terry scorer used by TRACE.

This is intentionally independent from the D4RL TRACE package.  A D4RL scorer
checkpoint is not semantically compatible with Go2 trajectories even when the
network shape happens to match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from scripts.reinforcement_learning.rwm_trace.reference_summary import REFERENCE_FEATURE_NAMES
from scripts.reinforcement_learning.rwm_trace.trajectory import GO2_COMMAND_MODES


GO2_FEATURE_NAMES = (
    "simulator_return_per_step",
    "survival_fraction",
    "terminal_flag",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_trend_per_second",
    "action_norm_mean",
    "action_norm_std",
    "action_saturation_rate",
    "action_delta_norm_mean",
    "state_delta_norm_mean",
    "base_speed_mean",
    "command_vx_mean",
    "command_vy_mean",
    "command_yaw_mean",
    "command_linear_speed_mean",
    "command_linear_active",
    "command_yaw_active",
    *(f"command_mode_{mode}" for mode in GO2_COMMAND_MODES),
    "linear_velocity_projection_mean",
    "linear_velocity_realization_ratio_mean",
    "linear_velocity_realization_ratio_steady",
    "yaw_velocity_realization_ratio_mean",
    "yaw_velocity_realization_ratio_steady",
    "command_direction_violation_rate",
    "command_direction_correct_fraction",
    "command_projected_velocity_window",
    "command_displacement_realization_ratio",
    "yaw_velocity_window",
    "yaw_displacement_realization_ratio",
    "linear_tracking_error_mean",
    "linear_tracking_error_steady",
    "yaw_tracking_error_mean",
    "yaw_tracking_error_steady",
    "tilt_mean",
    "tilt_max",
    "joint_velocity_rms",
    "actuator_force_rms",
    "contact_fraction_mean",
    "contact_switch_rate",
    "contact_fraction_rr",
    "contact_fraction_rl",
    "contact_switch_rate_rr",
    "contact_switch_rate_rl",
    "foot_swing_rate_fr",
    "foot_swing_rate_fl",
    "foot_swing_rate_rr",
    "foot_swing_rate_rl",
    "longest_stance_fraction_rr",
    "longest_stance_fraction_rl",
    "rear_duty_factor_abs_difference",
    "rr_calf_relative_position_mean",
    "base_velocity_window_x",
    "base_velocity_window_y",
    "base_height_mean",
    "base_height_trend_per_second",
    "foot_height_max_rr",
    "foot_height_max_rl",
    "foot_height_range_fr",
    "foot_height_range_fl",
    "foot_height_range_rr",
    "foot_height_range_rl",
    "foot_speed_mean_rr",
    "foot_speed_mean_rl",
    "nonfinite_flag",
    "reset_reconstruction_error",
    *REFERENCE_FEATURE_NAMES,
)


@dataclass(frozen=True)
class FeatureStats:
    names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: list(value) for key, value in data.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureStats":
        return cls(
            names=tuple(str(value) for value in data["names"]),
            mean=tuple(float(value) for value in data["mean"]),
            std=tuple(float(value) for value in data["std"]),
        )


class Go2TraceScorer(nn.Module):
    """Small MLP whose score difference parameterizes Bradley-Terry labels."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _finite(value: Any) -> tuple[float, float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if not np.isfinite(parsed):
        return 0.0, 1.0
    return parsed, 0.0


def feature_matrix(
    summaries: Sequence[dict[str, Any]],
    names: Sequence[str] = GO2_FEATURE_NAMES,
) -> np.ndarray:
    """Return values followed by explicit missingness indicators."""

    rows: list[list[float]] = []
    for summary in summaries:
        parsed = [_finite(summary.get(name)) for name in names]
        rows.append([value for value, _ in parsed] + [missing for _, missing in parsed])
    if not rows:
        return np.zeros((0, 2 * len(names)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def fit_feature_stats(features: np.ndarray, names: Sequence[str] = GO2_FEATURE_NAMES) -> FeatureStats:
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    expanded_names = tuple(names) + tuple(f"{name}_missing" for name in names)
    return FeatureStats(
        names=expanded_names,
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
    )


def normalize_features(features: np.ndarray, stats: FeatureStats) -> np.ndarray:
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    if features.shape[-1] != len(mean):
        raise ValueError(f"Feature width {features.shape[-1]} does not match scorer width {len(mean)}.")
    return ((np.asarray(features, dtype=np.float32) - mean) / std).astype(np.float32)


def save_scorer_checkpoint(
    path: str,
    model: Go2TraceScorer,
    stats: FeatureStats,
    *,
    hidden_dim: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "format_version": "go2_trace_scorer_v10_v1",
            "feature_schema": "go2_trace_length_normalized_features_v10",
            "model_state_dict": model.state_dict(),
            "input_dim": len(stats.mean),
            "hidden_dim": int(hidden_dim),
            "feature_stats": stats.to_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_scorer_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[Go2TraceScorer, FeatureStats, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != "go2_trace_scorer_v10_v1":
        raise ValueError(f"Not a Go2 TRACE scorer checkpoint: {path}")
    if checkpoint.get("feature_schema") != "go2_trace_length_normalized_features_v10":
        raise ValueError(f"TRACE scorer uses an incompatible feature schema: {path}")
    stats = FeatureStats.from_dict(checkpoint["feature_stats"])
    model = Go2TraceScorer(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, stats, dict(checkpoint.get("metadata") or {})


def score_summaries(
    model: Go2TraceScorer,
    summaries: Sequence[dict[str, Any]],
    stats: FeatureStats,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    base_names = tuple(name for name in stats.names if not name.endswith("_missing"))
    features = normalize_features(feature_matrix(summaries, base_names), stats)
    with torch.no_grad():
        scores = model(torch.as_tensor(features, device=device, dtype=torch.float32))
    return scores.detach().cpu().numpy().astype(np.float32)
