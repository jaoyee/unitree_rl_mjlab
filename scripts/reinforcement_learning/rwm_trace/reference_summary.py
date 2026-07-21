"""Attach matched RWM-reference context to Go2 TRACE summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REFERENCE_SUMMARY_FIELDS = (
    "simulator_return_per_step",
    "survival_fraction",
    "terminal_flag",
    "linear_tracking_error_mean",
    "yaw_tracking_error_mean",
    "tilt_mean",
    "tilt_max",
    "action_saturation_rate",
    "action_delta_norm_mean",
    "contact_fraction_mean",
    "contact_switch_rate",
    "contact_fraction_rr",
    "contact_fraction_rl",
)

REFERENCE_FEATURE_NAMES = tuple(f"rwm_reference_{name}" for name in REFERENCE_SUMMARY_FIELDS) + (
    "rwm_reference_available",
    "delta_return_per_step_vs_rwm",
    "delta_survival_fraction_vs_rwm",
    "delta_terminal_flag_vs_rwm",
    "delta_linear_tracking_error_vs_rwm",
    "delta_yaw_tracking_error_vs_rwm",
    "delta_tilt_mean_vs_rwm",
    "delta_tilt_max_vs_rwm",
    "delta_action_saturation_rate_vs_rwm",
    "delta_action_delta_norm_mean_vs_rwm",
    "delta_contact_fraction_mean_vs_rwm",
    "delta_contact_switch_rate_vs_rwm",
    "delta_contact_fraction_rr_vs_rwm",
    "delta_contact_fraction_rl_vs_rwm",
)


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def start_key(summary: dict[str, Any]) -> str:
    return str(summary.get("start_state_key", summary.get("start_state_id", -1)))


def compact_reference_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary.get(key)
        for key in (
            "trajectory_id",
            "start_state_id",
            "start_state_key",
            "source_kind",
            *REFERENCE_SUMMARY_FIELDS,
            "reference_history_padded",
            "reference_history_valid_length",
        )
        if key in summary
    }


def attach_reference_features(
    summary: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(summary)
    if reference is None:
        out["rwm_reference_available"] = 0.0
        return out

    out["rwm_reference_available"] = 1.0
    out["rwm_reference_summary"] = compact_reference_summary(reference)
    for field in REFERENCE_SUMMARY_FIELDS:
        ref_value = _as_float(reference.get(field))
        sim_value = _as_float(summary.get(field))
        out[f"rwm_reference_{field}"] = ref_value
        if field == "terminal_flag":
            out["delta_terminal_flag_vs_rwm"] = ref_value - sim_value
        elif field == "simulator_return_per_step":
            out["delta_return_per_step_vs_rwm"] = sim_value - ref_value
        elif field == "survival_fraction":
            out["delta_survival_fraction_vs_rwm"] = sim_value - ref_value
        elif field in {
            "linear_tracking_error_mean",
            "yaw_tracking_error_mean",
            "tilt_mean",
            "tilt_max",
            "action_saturation_rate",
            "action_delta_norm_mean",
        }:
            out[f"delta_{field}_vs_rwm"] = ref_value - sim_value
        elif field in {
            "contact_fraction_mean",
            "contact_switch_rate",
            "contact_fraction_rr",
            "contact_fraction_rl",
        }:
            out[f"delta_{field}_vs_rwm"] = sim_value - ref_value
    return out


def load_reference_map(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = start_key(row)
        if key in mapping:
            raise ValueError(f"Duplicate RWM reference start_state_key={key!r} in {path}")
        mapping[key] = row
    return mapping


def attach_references_to_summaries(
    summaries: Iterable[dict[str, Any]],
    references: dict[str, dict[str, Any]] | None,
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for summary in summaries:
        key = start_key(summary)
        reference = None if references is None else references.get(key)
        if reference is None and strict:
            missing.append(key)
        out.append(attach_reference_features(summary, reference))
    if missing:
        unique = sorted(set(missing))
        raise ValueError(
            f"Missing matched RWM reference summaries for {len(unique)} start keys; "
            f"examples={unique[:10]}"
        )
    return out
