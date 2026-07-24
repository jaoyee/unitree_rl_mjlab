"""Configuration contracts for the portable TRACE pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _absolute_without_resolving_symlinks(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _required(document: dict[str, Any], name: str) -> Any:
    value = document.get(name)
    if value in (None, ""):
        raise ValueError(f"Missing required configuration field: {name}")
    return value


@dataclass(frozen=True)
class BaselineAdapter:
    """The only information TRACE is allowed to obtain from an RWM baseline."""

    name: str
    setting: str
    source_dataset: Path
    rollout_actor_checkpoint: Path
    reward_config: Path
    world_model_checkpoint: Path | None
    candidate_simulator: str
    candidate_reset_mode: str
    replay_zero_observation_indices: tuple[int, ...]
    replay_reward_version: str | None
    replay_recompute_reward_after_selection: bool
    training_initial_checkpoint: Path | None
    training_command: tuple[str, ...]
    training_environment: dict[str, str]

    @classmethod
    def from_document(cls, document: dict[str, Any], base: Path) -> "BaselineAdapter":
        if document.get("schema") != "trace_baseline_adapter_v1":
            raise ValueError("Baseline adapter schema must be trace_baseline_adapter_v1.")
        training = dict(document.get("training") or {})
        candidate = dict(document.get("candidate") or {})
        replay = dict(document.get("replay") or {})
        setting = str(document.get("setting", "sim"))
        if setting not in {"sim", "real"}:
            raise ValueError("Baseline adapter setting must be sim or real.")
        candidate_simulator = str(candidate.get("simulator", "normal"))
        if candidate_simulator != "normal":
            raise ValueError(
                "Portable TRACE candidates must be collected in the normal simulator."
            )
        expected_reset_mode = (
            "exact_snapshot" if setting == "sim" else "canonical_real_projection"
        )
        candidate_reset_mode = str(
            candidate.get("reset_mode", expected_reset_mode)
        )
        if candidate_reset_mode != expected_reset_mode:
            raise ValueError(
                f"Portable TRACE requires candidate.reset_mode="
                f"{expected_reset_mode} for setting={setting}."
            )
        recompute_reward = bool(
            replay.get("recompute_reward_after_selection", True)
        )
        if not recompute_reward:
            raise ValueError(
                "TRACE replay reward must be recomputed after selection."
            )
        zero_indices = tuple(
            int(value) for value in replay.get("zero_observation_indices", [])
        )
        if any(index < 0 for index in zero_indices):
            raise ValueError("Replay zero_observation_indices must be non-negative.")
        if len(zero_indices) != len(set(zero_indices)):
            raise ValueError("Replay zero_observation_indices must be unique.")
        return cls(
            name=str(_required(document, "name")),
            setting=setting,
            source_dataset=_resolve(base, str(_required(document, "source_dataset"))),
            rollout_actor_checkpoint=_resolve(
                base, str(_required(document, "rollout_actor_checkpoint"))
            ),
            reward_config=_resolve(base, str(_required(document, "reward_config"))),
            world_model_checkpoint=_resolve(base, document.get("world_model_checkpoint")),
            candidate_simulator=candidate_simulator,
            candidate_reset_mode=candidate_reset_mode,
            replay_zero_observation_indices=zero_indices,
            replay_reward_version=(
                str(replay["reward_version"])
                if replay.get("reward_version") is not None
                else None
            ),
            replay_recompute_reward_after_selection=recompute_reward,
            training_initial_checkpoint=_resolve(
                base, training.get("initial_checkpoint")
            ),
            training_command=tuple(str(item) for item in training.get("command", [])),
            training_environment={
                str(key): str(value)
                for key, value in dict(training.get("environment") or {}).items()
            },
        )


@dataclass(frozen=True)
class PipelineConfig:
    repo_root: Path
    python: Path
    protocol: Path
    selection_backend: str
    scorer_checkpoint: Path | None
    output_root: Path
    refresh_cycle: int
    seed: int
    simulator: dict[str, Any]
    tools: dict[str, Path]

    @classmethod
    def from_document(cls, document: dict[str, Any], base: Path) -> "PipelineConfig":
        if document.get("schema") != "portable_trace_pipeline_v1":
            raise ValueError("Pipeline schema must be portable_trace_pipeline_v1.")
        repo_root = _resolve(base, str(_required(document, "repo_root")))
        tools = {
            str(name): _resolve(repo_root, str(path))
            for name, path in dict(_required(document, "tools")).items()
        }
        required_tools = {
            "source_selector",
            "candidate_collector",
            "candidate_merger",
            "candidate_auditor",
            "summary_replay_builder",
            "distribution_selector",
            "replay_manifest",
        }
        missing = sorted(required_tools - tools.keys())
        if missing:
            raise ValueError(f"Pipeline tool mapping is incomplete: {missing}")
        selection = dict(document.get("selection") or {})
        selection_backend = str(selection.get("backend", "scorer"))
        if selection_backend not in {"rule", "scorer"}:
            raise ValueError("Selection backend must be rule or scorer.")
        scorer_value = selection.get(
            "scorer_checkpoint", document.get("scorer_checkpoint")
        )
        if selection_backend == "scorer" and not scorer_value:
            raise ValueError(
                "A scorer checkpoint is required when selection.backend=scorer."
            )
        return cls(
            repo_root=repo_root,
            # Resolving the final symlink of a virtualenv Python turns it into
            # the bare base interpreter and silently drops the venv packages.
            python=_absolute_without_resolving_symlinks(
                repo_root, str(_required(document, "python"))
            ),
            protocol=_resolve(repo_root, str(_required(document, "protocol"))),
            selection_backend=selection_backend,
            scorer_checkpoint=(
                _resolve(repo_root, str(scorer_value))
                if scorer_value is not None
                else None
            ),
            output_root=_resolve(base, str(_required(document, "output_root"))),
            refresh_cycle=int(document.get("refresh_cycle", 1)),
            seed=int(document.get("seed", 42)),
            simulator=dict(_required(document, "simulator")),
            tools=tools,
        )


def load_configuration(
    pipeline_path: str | Path, baseline_path: str | Path
) -> tuple[PipelineConfig, BaselineAdapter]:
    pipeline_file = Path(pipeline_path).expanduser().resolve()
    baseline_file = Path(baseline_path).expanduser().resolve()
    pipeline = PipelineConfig.from_document(
        json.loads(pipeline_file.read_text(encoding="utf-8")),
        pipeline_file.parent,
    )
    baseline = BaselineAdapter.from_document(
        json.loads(baseline_file.read_text(encoding="utf-8")),
        baseline_file.parent,
    )
    return pipeline, baseline
