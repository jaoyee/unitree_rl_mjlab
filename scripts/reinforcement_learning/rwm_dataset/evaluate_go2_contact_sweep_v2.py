"""Compare contact thresholds and their coupled base-velocity estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--threshold", type=float, action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--stand-epsilon", type=float, default=1e-3)
    return parser.parse_args()


def stack(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.stack(value)


def corr(a: torch.Tensor, b: torch.Tensor) -> float | None:
    valid = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[valid].float(), b[valid].float()
    if a.numel() < 2 or float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return None
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


def summarize(path: Path, stand_epsilon: float) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    states = stack(data["states"]).reshape(-1, 45).float()
    commands = stack(data["commands"]).reshape(-1, 3).float()
    contacts = stack(data["contacts"]).reshape(-1, 4).float()
    confidence = stack(data["base_lin_vel_confidence"]).reshape(-1).float()
    terminations = stack(data["terminations"]).reshape(-1).float()
    force = stack(data["foot_forces"]).reshape(-1, 4).float()
    velocity = states[:, :3]
    stand = commands.abs().amax(dim=1) <= stand_epsilon
    moving = ~stand
    per_axis_corr = [corr(velocity[moving, i], commands[moving, i]) for i in range(3)]
    stand_abs = velocity[stand].abs() if bool(stand.any()) else torch.empty((0, 3))
    return {
        "transitions": len(states),
        "termination_count": int((terminations > 0.5).sum()),
        "contact_fraction_per_foot": contacts.mean(0).tolist(),
        "all_four_contact_fraction": float((contacts.sum(1) == 4).float().mean()),
        "zero_contact_fraction": float((contacts.sum(1) == 0).float().mean()),
        "contact_count_histogram": {
            str(count): int((contacts.sum(1) == count).sum()) for count in range(5)
        },
        "foot_force_quantiles": {
            str(q): torch.quantile(force, q, dim=0).tolist() for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "base_velocity_command_correlation": per_axis_corr,
        "base_velocity_confidence_mean": float(confidence.mean()),
        "base_velocity_confidence_nonzero_fraction": float((confidence > 0).float().mean()),
        "base_velocity_abs_mean": velocity.abs().mean(0).tolist(),
        "base_velocity_abs_p95": torch.quantile(velocity.abs(), 0.95, dim=0).tolist(),
        "stand_count": int(stand.sum()),
        "stand_base_velocity_abs_mean": stand_abs.mean(0).tolist() if stand_abs.numel() else None,
        "stand_base_velocity_abs_p95": (
            torch.quantile(stand_abs, 0.95, dim=0).tolist() if stand_abs.numel() else None
        ),
        "metadata": data.get("metadata", {}),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.sweep_root).expanduser()
    conditions = args.condition or sorted(path.name for path in root.iterdir() if path.is_dir())
    thresholds = args.threshold or [10.0, 12.0, 15.0, 18.0]
    report: dict[str, Any] = {
        "schema": "go2_contact_sweep_v2",
        "stand_epsilon": float(args.stand_epsilon),
        "conditions": {},
    }
    for condition in conditions:
        report["conditions"][condition] = {}
        for threshold in thresholds:
            label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
            path = root / condition / f"threshold_{label}" / "dataset_all.pt"
            report["conditions"][condition][label] = summarize(path, float(args.stand_epsilon))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    compact = {
        condition: {
            threshold: {
                "all4": values["all_four_contact_fraction"],
                "zero": values["zero_contact_fraction"],
                "corr": values["base_velocity_command_correlation"],
                "confidence": values["base_velocity_confidence_mean"],
                "stand_abs_mean": values["stand_base_velocity_abs_mean"],
            }
            for threshold, values in rows.items()
        }
        for condition, rows in report["conditions"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
