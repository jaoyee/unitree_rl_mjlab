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
    training_initial_checkpoint: Path | None
    training_command: tuple[str, ...]
    training_environment: dict[str, str]

    @classmethod
    def from_document(cls, document: dict[str, Any], base: Path) -> "BaselineAdapter":
        if document.get("schema") != "trace_baseline_adapter_v1":
            raise ValueError("Baseline adapter schema must be trace_baseline_adapter_v1.")
        training = dict(document.get("training") or {})
        setting = str(document.get("setting", "sim"))
        if setting not in {"sim", "real"}:
            raise ValueError("Baseline adapter setting must be sim or real.")
        return cls(
            name=str(_required(document, "name")),
            setting=setting,
            source_dataset=_resolve(base, str(_required(document, "source_dataset"))),
            rollout_actor_checkpoint=_resolve(
                base, str(_required(document, "rollout_actor_checkpoint"))
            ),
            reward_config=_resolve(base, str(_required(document, "reward_config"))),
            world_model_checkpoint=_resolve(base, document.get("world_model_checkpoint")),
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
    scorer_checkpoint: Path
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
        return cls(
            repo_root=repo_root,
            # Resolving the final symlink of a virtualenv Python turns it into
            # the bare base interpreter and silently drops the venv packages.
            python=_absolute_without_resolving_symlinks(
                repo_root, str(_required(document, "python"))
            ),
            protocol=_resolve(repo_root, str(_required(document, "protocol"))),
            scorer_checkpoint=_resolve(
                repo_root, str(_required(document, "scorer_checkpoint"))
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
