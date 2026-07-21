"""Deterministic global metric selection for the scorer-free V10 pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path


METRIC_SCHEMA = "go2_trace_v10_metric_pilot_v1"


def load_metric_config(
    path: str | Path,
    *,
    base_protocol_sha256: str,
) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).expanduser().resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != METRIC_SCHEMA:
        raise ValueError(f"Unsupported metric-pilot config schema: {config.get('schema')!r}")
    if config.get("base_protocol_sha256") != base_protocol_sha256:
        raise ValueError("Metric-pilot config was not built for the active V10 protocol SHA.")
    selection = dict(config.get("selection") or {})
    if selection.get("scope") != "global":
        raise ValueError("Metric-pilot selection must be global.")
    if int(selection.get("selected_trajectory_count", 0)) <= 0:
        raise ValueError("Metric-pilot selected_trajectory_count must be positive.")
    tracking_weight = float(selection.get("tracking_weight", -1.0))
    stability_weight = float(selection.get("stability_weight", -1.0))
    if not np.isclose(tracking_weight + stability_weight, 1.0):
        raise ValueError("Metric-pilot tracking and stability weights must sum to one.")
    if min(tracking_weight, stability_weight) < 0.0:
        raise ValueError("Metric-pilot weights must be non-negative.")
    if selection.get("sampling_strategy") != "uniform_selected_replay":
        raise ValueError("Metric-pilot replay must sample uniformly from the globally selected set.")
    return config, resolved, sha256_path(resolved)


def _average_percentile_quality(values: np.ndarray, *, smaller_is_better: bool) -> np.ndarray:
    """Return tie-aware percentile quality in [0, 1]."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Metric values must be a finite non-empty vector.")
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(np.concatenate(([0], counts[:-1]))).astype(np.float64)
    average_ranks = starts + (counts.astype(np.float64) - 1.0) / 2.0
    percentiles = average_ranks[inverse] / max(len(values) - 1, 1)
    return 1.0 - percentiles if smaller_is_better else percentiles


def _finite_field(
    summaries: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    field: str,
) -> np.ndarray:
    values = np.asarray([float(summaries[int(index)].get(field, np.nan)) for index in indices])
    if not np.isfinite(values).all():
        bad = indices[~np.isfinite(values)][:10].tolist()
        raise ValueError(f"Metric field {field!r} is missing/non-finite for eligible trajectories {bad}.")
    return values


def metric_scores(
    summaries: Sequence[dict[str, Any]],
    eligible_indices: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute global tracking/stability quality without command-mode stratification."""

    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if len(eligible) == 0:
        raise ValueError("Metric selection has no eligible trajectories.")
    selection = dict(config["selection"])
    tracking_cfg = dict(config["tracking"])
    stability_cfg = dict(config["stability"])

    linear_error = _finite_field(summaries, eligible, str(tracking_cfg["linear_field"]))
    yaw_error = _finite_field(summaries, eligible, str(tracking_cfg["yaw_field"]))
    command_linear = np.asarray(
        [
            np.hypot(
                float(summaries[int(index)].get("command_vx_mean", np.nan)),
                float(summaries[int(index)].get("command_vy_mean", np.nan)),
            )
            for index in eligible
        ],
        dtype=np.float64,
    )
    command_yaw = np.asarray(
        [abs(float(summaries[int(index)].get("command_yaw_mean", np.nan))) for index in eligible],
        dtype=np.float64,
    )
    if not np.isfinite(command_linear).all() or not np.isfinite(command_yaw).all():
        raise ValueError("Metric selection requires finite command means.")
    linear_scale = np.maximum(command_linear, float(selection["inactive_linear_scale"]))
    yaw_scale = np.maximum(command_yaw, float(selection["inactive_yaw_scale"]))
    normalized_linear_error = linear_error / linear_scale
    normalized_yaw_error = yaw_error / yaw_scale
    tracking_cost = 0.5 * (normalized_linear_error + normalized_yaw_error)
    tracking_quality = _average_percentile_quality(tracking_cost, smaller_is_better=True)

    stability_components: list[np.ndarray] = []
    component_quality: dict[str, np.ndarray] = {}
    for field in stability_cfg["cost_fields"]:
        quality = _average_percentile_quality(
            _finite_field(summaries, eligible, str(field)), smaller_is_better=True
        )
        component_quality[str(field)] = quality
        stability_components.append(quality)
    for field in stability_cfg["quality_fields"]:
        quality = _average_percentile_quality(
            _finite_field(summaries, eligible, str(field)), smaller_is_better=False
        )
        component_quality[str(field)] = quality
        stability_components.append(quality)
    stability_quality = np.mean(np.stack(stability_components, axis=0), axis=0)
    combined = (
        float(selection["tracking_weight"]) * tracking_quality
        + float(selection["stability_weight"]) * stability_quality
    )
    if not np.isfinite(combined).all():
        raise ValueError("Metric selection produced non-finite scores.")
    return combined, {
        "tracking_cost": tracking_cost,
        "normalized_linear_tracking_error": normalized_linear_error,
        "normalized_yaw_tracking_error": normalized_yaw_error,
        "tracking_quality": tracking_quality,
        "stability_quality": stability_quality,
        **{f"stability_quality_{key}": value for key, value in component_quality.items()},
    }


def select_global_metric_pilot(
    summaries: Sequence[dict[str, Any]],
    eligible_indices: np.ndarray,
    *,
    config: Mapping[str, Any],
    selection: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select an exact global top set using metrics or a global-random control."""

    eligible = np.asarray(eligible_indices, dtype=np.int64)
    target = int(config["selection"]["selected_trajectory_count"])
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("Eligible trajectory indices must be unique.")
    if len(eligible) < target:
        raise ValueError(
            f"Global metric pilot requires {target} valid candidates, but only {len(eligible)} remain."
        )
    full_scores = np.zeros(len(summaries), dtype=np.float64)
    score_components: dict[str, np.ndarray] = {}
    if selection == "metric":
        eligible_scores, score_components = metric_scores(summaries, eligible, config)
    elif selection == "random":
        eligible_scores = np.random.default_rng(int(seed)).random(len(eligible))
    else:
        raise ValueError(f"Unsupported global metric-pilot selection: {selection!r}")
    full_scores[eligible] = eligible_scores
    order = np.lexsort((eligible, -eligible_scores))
    selected = eligible[order[:target]]
    selected_local_mask = np.isin(eligible, selected)
    for local_index, global_index in enumerate(eligible):
        summary = summaries[int(global_index)]
        summary["metric_pilot_score"] = float(eligible_scores[local_index])
        for name, values in score_components.items():
            summary[f"metric_pilot_{name}"] = float(values[local_index])
    diagnostics = {
        "selection_protocol": f"unconstrained_global_{selection}_top25_v1",
        "eligible_count": int(len(eligible)),
        "selected_count": int(len(selected)),
        "selection_ratio_of_summarized_candidates": float(len(selected) / max(len(summaries), 1)),
        "selection_ratio_of_eligible_candidates": float(len(selected) / len(eligible)),
        "score_min": float(np.min(eligible_scores[selected_local_mask])),
        "score_mean": float(np.mean(eligible_scores[selected_local_mask])),
        "score_max": float(np.max(eligible_scores[selected_local_mask])),
        "command_distribution_constrained": False,
        "per_start_minimum": None,
        "per_start_maximum": None,
    }
    return selected.astype(np.int64), full_scores, diagnostics
