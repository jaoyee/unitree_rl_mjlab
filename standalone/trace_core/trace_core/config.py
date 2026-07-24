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
    schema: str
    setting: str
    source_dataset: Path
    rollout_actor_checkpoint: Path
    reward_config: Path
    world_model_checkpoint: Path | None
    candidate_simulator: str
    candidate_reset_mode: str
    candidate_resample_groups: int
    candidate_rollout_policy_source: str
    observation_full_dim: int | None
    actor_observation_dim: int | None
    critic_observation_dim: int | None
    actor_excluded_observation_indices: tuple[int, ...]
    unsupervised_world_model_output_indices: tuple[int, ...]
    replay_observation_policy: str
    source_leakage_audit_required: bool
    replay_zero_observation_indices: tuple[int, ...]
    replay_reward_version: str | None
    replay_recompute_reward_after_selection: bool
    replay_n_step: int | None
    replay_gamma: float | None
    replay_terminal_reward_override: bool
    replay_epistemic_uncertainty_mode: str
    replay_additional_reward_penalties: bool
    training_policy_initialization: str
    training_initial_checkpoint: Path | None
    training_command: tuple[str, ...]
    training_environment: dict[str, str]

    @classmethod
    def from_document(cls, document: dict[str, Any], base: Path) -> "BaselineAdapter":
        schema = str(document.get("schema"))
        if schema not in {"trace_baseline_adapter_v1", "trace_baseline_adapter_v2"}:
            raise ValueError(
                "Baseline adapter schema must be trace_baseline_adapter_v1 or "
                "trace_baseline_adapter_v2."
            )
        training = dict(document.get("training") or {})
        candidate = dict(document.get("candidate") or {})
        observation = dict(document.get("observation") or {})
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
        candidate_resample_groups = int(candidate.get("resample_groups", 1))
        if candidate_resample_groups < 1:
            raise ValueError("candidate.resample_groups must be positive.")
        candidate_rollout_policy_source = str(
            candidate.get(
                "rollout_policy_source",
                "baseline_actor_checkpoint",
            )
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
        observation_full_dim = (
            int(observation["full_dim"])
            if observation.get("full_dim") is not None
            else None
        )
        actor_observation_dim = (
            int(observation["actor_dim"])
            if observation.get("actor_dim") is not None
            else None
        )
        critic_observation_dim = (
            int(observation["critic_dim"])
            if observation.get("critic_dim") is not None
            else None
        )
        actor_excluded_indices = tuple(
            int(value)
            for value in observation.get("actor_excluded_indices", [])
        )
        unsupervised_output_indices = tuple(
            int(value)
            for value in observation.get(
                "unsupervised_world_model_output_indices", []
            )
        )
        for name, indices in {
            "observation.actor_excluded_indices": actor_excluded_indices,
            "observation.unsupervised_world_model_output_indices": (
                unsupervised_output_indices
            ),
        }.items():
            if any(index < 0 for index in indices):
                raise ValueError(f"{name} must be non-negative.")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{name} must be unique.")
        replay_observation_policy = str(
            replay.get("observation_policy", "explicit_zero_indices")
        )
        if replay_observation_policy not in {
            "explicit_zero_indices",
            "source_native_full_state",
            "frozen_rwm_projection",
        }:
            raise ValueError("Unsupported replay.observation_policy.")
        source_leakage_audit_required = bool(
            replay.get("source_leakage_audit_required", False)
        )
        replay_n_step = (
            int(replay["n_step"]) if replay.get("n_step") is not None else None
        )
        replay_gamma = (
            float(replay["gamma"]) if replay.get("gamma") is not None else None
        )
        terminal_reward_override = bool(
            replay.get("terminal_reward_override", False)
        )
        epistemic_uncertainty_mode = str(
            replay.get("epistemic_uncertainty_mode", "unspecified")
        )
        additional_reward_penalties = bool(
            replay.get("additional_reward_penalties", False)
        )
        policy_initialization = str(
            training.get(
                "policy_initialization",
                "resume" if training.get("initial_checkpoint") else "from_zero",
            )
        )
        if policy_initialization not in {"from_zero", "resume"}:
            raise ValueError(
                "training.policy_initialization must be from_zero or resume."
            )
        training_initial_checkpoint = _resolve(
            base, training.get("initial_checkpoint")
        )
        if schema == "trace_baseline_adapter_v2":
            if candidate_rollout_policy_source != "baseline_actor_checkpoint":
                raise ValueError(
                    "V2 candidate rollout must use the selected baseline actor "
                    "checkpoint; expert-policy rollout is forbidden."
                )
            if candidate_resample_groups < 5:
                raise ValueError(
                    "V2 TRACE requires at least five independently resampled "
                    "candidate groups."
                )
            if None in {
                observation_full_dim,
                actor_observation_dim,
                critic_observation_dim,
            }:
                raise ValueError(
                    "V2 adapters require observation full_dim, actor_dim and "
                    "critic_dim."
                )
            assert observation_full_dim is not None
            assert actor_observation_dim is not None
            assert critic_observation_dim is not None
            if min(
                observation_full_dim,
                actor_observation_dim,
                critic_observation_dim,
            ) < 1:
                raise ValueError("V2 observation dimensions must be positive.")
            for name, indices in {
                "observation.actor_excluded_indices": actor_excluded_indices,
                "observation.unsupervised_world_model_output_indices": (
                    unsupervised_output_indices
                ),
                "replay.zero_observation_indices": zero_indices,
            }.items():
                if any(index >= observation_full_dim for index in indices):
                    raise ValueError(
                        f"{name} contains an index outside observation.full_dim."
                    )
            if actor_observation_dim != (
                observation_full_dim - len(actor_excluded_indices)
            ):
                raise ValueError(
                    "observation.actor_dim does not match full_dim minus "
                    "actor_excluded_indices."
                )
            if critic_observation_dim != observation_full_dim:
                raise ValueError(
                    "V2 TRACE currently requires critic_dim == full_dim."
                )
            if not set(unsupervised_output_indices).issubset(
                actor_excluded_indices
            ):
                raise ValueError(
                    "Unsupervised world-model outputs must be excluded from "
                    "the actor observation."
                )
            if replay_observation_policy == "explicit_zero_indices":
                if not zero_indices:
                    raise ValueError(
                        "explicit_zero_indices requires non-empty "
                        "replay.zero_observation_indices."
                    )
            elif zero_indices:
                raise ValueError(
                    "zero_observation_indices must be empty unless "
                    "observation_policy=explicit_zero_indices."
                )
            critic_sees_unsupervised = bool(unsupervised_output_indices)
            if (
                critic_sees_unsupervised
                and replay_observation_policy == "source_native_full_state"
                and not source_leakage_audit_required
            ):
                raise ValueError(
                    "source_native_full_state with unsupervised RWM outputs "
                    "requires replay.source_leakage_audit_required=true."
                )
            if not replay.get("reward_version"):
                raise ValueError("V2 adapters require replay.reward_version.")
            if replay_n_step is None or replay_n_step < 1:
                raise ValueError("V2 adapters require positive replay.n_step.")
            if replay_gamma is None or not 0.0 < replay_gamma <= 1.0:
                raise ValueError("V2 adapters require replay.gamma in (0, 1].")
            if terminal_reward_override:
                raise ValueError(
                    "V2 TRACE must not override the frozen baseline terminal reward."
                )
            if epistemic_uncertainty_mode != "physical_zero":
                raise ValueError(
                    "V2 replay reward requires "
                    "replay.epistemic_uncertainty_mode=physical_zero."
                )
            if additional_reward_penalties:
                raise ValueError(
                    "V2 TRACE must not add reward penalties absent from the "
                    "selected RWM baseline."
                )
            if policy_initialization != "from_zero":
                raise ValueError(
                    "V2 TRACE policy and critic must initialize from zero."
                )
            if training_initial_checkpoint is not None:
                raise ValueError(
                    "from_zero training forbids training.initial_checkpoint."
                )
        return cls(
            name=str(_required(document, "name")),
            schema=schema,
            setting=setting,
            source_dataset=_resolve(base, str(_required(document, "source_dataset"))),
            rollout_actor_checkpoint=_resolve(
                base, str(_required(document, "rollout_actor_checkpoint"))
            ),
            reward_config=_resolve(base, str(_required(document, "reward_config"))),
            world_model_checkpoint=_resolve(base, document.get("world_model_checkpoint")),
            candidate_simulator=candidate_simulator,
            candidate_reset_mode=candidate_reset_mode,
            candidate_resample_groups=candidate_resample_groups,
            candidate_rollout_policy_source=candidate_rollout_policy_source,
            observation_full_dim=observation_full_dim,
            actor_observation_dim=actor_observation_dim,
            critic_observation_dim=critic_observation_dim,
            actor_excluded_observation_indices=actor_excluded_indices,
            unsupervised_world_model_output_indices=unsupervised_output_indices,
            replay_observation_policy=replay_observation_policy,
            source_leakage_audit_required=source_leakage_audit_required,
            replay_zero_observation_indices=zero_indices,
            replay_reward_version=(
                str(replay["reward_version"])
                if replay.get("reward_version") is not None
                else None
            ),
            replay_recompute_reward_after_selection=recompute_reward,
            replay_n_step=replay_n_step,
            replay_gamma=replay_gamma,
            replay_terminal_reward_override=terminal_reward_override,
            replay_epistemic_uncertainty_mode=epistemic_uncertainty_mode,
            replay_additional_reward_penalties=additional_reward_penalties,
            training_policy_initialization=policy_initialization,
            training_initial_checkpoint=training_initial_checkpoint,
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
    candidate_group_index: int
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
            candidate_group_index=int(document.get("candidate_group_index", 0)),
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
