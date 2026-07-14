"""Tune simple, interpretable Go2 contact-velocity fusion on simulator data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--validation-npz", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _flatten(array: np.ndarray) -> np.ndarray:
    return array.reshape(-1, *array.shape[2:])


def _metrics(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    error = prediction[valid] - truth[valid]
    return {
        "valid_fraction": float(valid.mean()),
        "mae_xyz": np.mean(np.abs(error), axis=0).tolist(),
        "rmse_xyz": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "mae_xy": float(np.linalg.norm(error[:, :2], axis=1).mean()),
        "rmse_xy": float(np.sqrt(np.mean(np.sum(np.square(error[:, :2]), axis=1)))),
        "bias_xyz": np.mean(error, axis=0).tolist(),
    }


def _contact_median(per_foot: np.ndarray, contact: np.ndarray) -> np.ndarray:
    masked = np.where(contact[..., None] > 0.5, per_foot, np.nan)
    return np.nanmedian(masked, axis=-2)


def _consensus(per_foot: np.ndarray, contact: np.ndarray) -> np.ndarray:
    median = _contact_median(per_foot, contact)
    residual = np.linalg.norm(per_foot - median[..., None, :], axis=-1)
    residual = np.where(contact > 0.5, residual, np.inf)
    # Keep every contacted foot within 0.12 m/s of the contact median.  If all
    # are rejected, fall back to the closest contacted foot.
    keep = (contact > 0.5) & (residual <= 0.12)
    no_keep = ~keep.any(axis=-1)
    closest = np.argmin(residual, axis=-1)
    if no_keep.any():
        rows = np.nonzero(no_keep)
        keep[rows + (closest[rows],)] = True
    weights = keep.astype(np.float32)
    return (per_foot * weights[..., None]).sum(axis=-2) / weights.sum(axis=-1, keepdims=True)


def _fit_affine(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.concatenate(
        (prediction[valid], np.ones((int(valid.sum()), 1), dtype=prediction.dtype)), axis=1
    )
    y = truth[valid]
    ridge = 1e-4 * np.eye(x.shape[1], dtype=x.dtype)
    ridge[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def _apply_affine(prediction: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return prediction @ coefficients[:3] + coefficients[3]


def _ema(sequence: np.ndarray, alpha: float) -> np.ndarray:
    output = np.empty_like(sequence)
    output[0] = sequence[0]
    for index in range(1, sequence.shape[0]):
        output[index] = alpha * sequence[index] + (1.0 - alpha) * output[index - 1]
    return output


def _load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    args = _parse_args()
    train = _load(args.train_npz)
    validation = _load(args.validation_npz)
    report: dict[str, object] = {"candidates": {}}

    train_valid = train["confidence"].reshape(-1) > 0.0
    validation_valid = validation["confidence"].reshape(-1) > 0.0
    train_truth = _flatten(train["truth"])
    validation_truth = _flatten(validation["truth"])

    train_candidates = {
        "contact_mean": _flatten(train["estimated"]),
        "contact_median": _flatten(_contact_median(train["per_foot"], train["contact"])),
        "contact_consensus": _flatten(_consensus(train["per_foot"], train["contact"])),
    }
    validation_candidates = {
        "contact_mean": _flatten(validation["estimated"]),
        "contact_median": _flatten(_contact_median(validation["per_foot"], validation["contact"])),
        "contact_consensus": _flatten(_consensus(validation["per_foot"], validation["contact"])),
    }

    for name in train_candidates:
        coefficients = _fit_affine(train_candidates[name], train_truth, train_valid)
        calibrated = _apply_affine(validation_candidates[name], coefficients)
        report["candidates"][name] = {
            "raw_validation": _metrics(
                validation_candidates[name], validation_truth, validation_valid
            ),
            "affine_coefficients": coefficients.tolist(),
            "affine_validation": _metrics(calibrated, validation_truth, validation_valid),
        }

        raw_sequence = validation_candidates[name].reshape(validation["truth"].shape)
        calibrated_sequence = _apply_affine(
            validation_candidates[name].reshape(validation["truth"].shape), coefficients
        )
        raw_ema_results = {}
        ema_results = {}
        for alpha in (0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
            raw_filtered = _ema(raw_sequence, alpha).reshape(-1, 3)
            filtered = _ema(calibrated_sequence, alpha).reshape(-1, 3)
            raw_ema_results[str(alpha)] = _metrics(
                raw_filtered, validation_truth, validation_valid
            )
            ema_results[str(alpha)] = _metrics(filtered, validation_truth, validation_valid)
        report["candidates"][name]["raw_ema_validation"] = raw_ema_results
        report["candidates"][name]["affine_ema_validation"] = ema_results

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
