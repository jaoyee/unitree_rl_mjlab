"""Portable TRACE scorer with no RWM, reward, actor, critic, or replay dependency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


FORBIDDEN_REWARD_FEATURES = frozenset(
    {
        "simulator_return_per_step",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_trend_per_second",
    }
)
SUPPORTED_FORMATS = frozenset({"go2_trace_scorer_v10_v1"})
SUPPORTED_FEATURE_SCHEMAS = frozenset({"go2_trace_length_normalized_features_v10"})


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _TraceScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
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


@dataclass(frozen=True)
class ScorerDescriptor:
    checkpoint_sha256: str
    format_version: str
    feature_schema: str
    base_feature_names: tuple[str, ...]
    input_dim: int
    hidden_dim: int
    scorer_profile: str | None
    objective_profile: str | None
    objective_weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "format_version": self.format_version,
            "feature_schema": self.feature_schema,
            "base_feature_names": list(self.base_feature_names),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "scorer_profile": self.scorer_profile,
            "objective_profile": self.objective_profile,
            "objective_weights": self.objective_weights,
        }


class PortableTraceScorer:
    """Score summary dictionaries using only checkpoint-contained schema/statistics."""

    def __init__(
        self,
        model: nn.Module,
        *,
        base_feature_names: Sequence[str],
        mean: Sequence[float],
        std: Sequence[float],
        descriptor: ScorerDescriptor,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.base_feature_names = tuple(base_feature_names)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.descriptor = descriptor
        self.device = torch.device(device)

    @staticmethod
    def _finite(value: Any) -> tuple[float, float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0, 1.0
        if not np.isfinite(parsed):
            return 0.0, 1.0
        return parsed, 0.0

    def feature_matrix(
        self, summaries: Sequence[dict[str, Any]]
    ) -> tuple[np.ndarray, dict[str, int]]:
        rows: list[list[float]] = []
        missing_counts = {name: 0 for name in self.base_feature_names}
        for summary in summaries:
            parsed = [self._finite(summary.get(name)) for name in self.base_feature_names]
            for name, (_, missing) in zip(self.base_feature_names, parsed, strict=True):
                missing_counts[name] += int(missing)
            rows.append(
                [value for value, _ in parsed]
                + [missing for _, missing in parsed]
            )
        if not rows:
            return np.zeros((0, len(self.mean)), dtype=np.float32), missing_counts
        features = np.asarray(rows, dtype=np.float32)
        if features.shape[1] != len(self.mean):
            raise ValueError(
                f"Summary feature width {features.shape[1]} differs from scorer input {len(self.mean)}."
            )
        normalized = ((features - self.mean) / self.std).astype(np.float32)
        return normalized, missing_counts

    def score(
        self,
        summaries: Sequence[dict[str, Any]],
        *,
        batch_size: int = 8192,
    ) -> tuple[np.ndarray, dict[str, int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        features, missing_counts = self.feature_matrix(summaries)
        output: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = torch.as_tensor(
                    features[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                output.append(self.model(batch).detach().cpu())
        scores = (
            torch.cat(output).numpy().astype(np.float32)
            if output
            else np.zeros(0, dtype=np.float32)
        )
        if not np.isfinite(scores).all():
            raise ValueError("Scorer produced non-finite values.")
        return scores, missing_counts


def _base_names(expanded_names: Sequence[str], input_dim: int) -> tuple[str, ...]:
    names = tuple(str(name) for name in expanded_names)
    if len(names) != input_dim or input_dim % 2:
        raise ValueError(
            f"Expected an even expanded feature schema of width {input_dim}, got {len(names)} names."
        )
    half = input_dim // 2
    base = names[:half]
    expected_missing = tuple(f"{name}_missing" for name in base)
    if names[half:] != expected_missing:
        raise ValueError("Checkpoint missingness features are not an exact suffix of base features.")
    return base


def load_scorer(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    reject_reward_features: bool = True,
) -> PortableTraceScorer:
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    format_version = str(checkpoint.get("format_version"))
    feature_schema = str(checkpoint.get("feature_schema"))
    if format_version not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported TRACE scorer format: {format_version!r}")
    if feature_schema not in SUPPORTED_FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported TRACE feature schema: {feature_schema!r}")
    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    stats = dict(checkpoint["feature_stats"])
    base_names = _base_names(stats["names"], input_dim)
    forbidden = sorted(set(base_names) & FORBIDDEN_REWARD_FEATURES)
    if reject_reward_features and forbidden:
        raise ValueError(
            "Scorer is coupled to baseline reward/return features and is not portable: "
            f"{forbidden}"
        )
    mean = tuple(float(value) for value in stats["mean"])
    std = tuple(float(value) for value in stats["std"])
    if len(mean) != input_dim or len(std) != input_dim:
        raise ValueError("Feature normalization width differs from checkpoint input_dim.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(np.asarray(std) <= 0):
        raise ValueError("Checkpoint contains invalid feature normalization statistics.")
    model = _TraceScorerMLP(input_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = dict(checkpoint.get("metadata") or {})
    raw_weights = dict(metadata.get("objective_weights") or {})
    descriptor = ScorerDescriptor(
        checkpoint_sha256=sha256_path(path),
        format_version=format_version,
        feature_schema=feature_schema,
        base_feature_names=base_names,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        scorer_profile=metadata.get("scorer_profile"),
        objective_profile=metadata.get("objective_profile"),
        objective_weights={str(key): float(value) for key, value in raw_weights.items()},
    )
    return PortableTraceScorer(
        model,
        base_feature_names=base_names,
        mean=mean,
        std=std,
        descriptor=descriptor,
        device=device,
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object.")
            rows.append(value)
    return rows
