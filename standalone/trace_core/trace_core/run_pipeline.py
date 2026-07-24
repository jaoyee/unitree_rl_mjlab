#!/usr/bin/env python3
"""Run the validated TRACE chain through a replaceable baseline adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import BaselineAdapter, PipelineConfig, load_configuration


STAGES = ("collect", "summarize", "score", "select", "replay", "train")


@dataclass(frozen=True)
class Command:
    stage: str
    argv: tuple[str, ...]
    environment: dict[str, str] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol(config: PipelineConfig) -> dict:
    return json.loads(config.protocol.read_text(encoding="utf-8"))


def _paths(config: PipelineConfig) -> dict[str, Path]:
    root = config.output_root
    return {
        "root": root,
        "source_ids": root / "candidates" / "source_ids.pt",
        "source_report": root / "candidates" / "source_ids_report.json",
        "source_batches": root / "candidates" / "source_batches",
        "candidate_parts": root / "candidates" / "candidate_parts",
        "candidate_dataset": root / "candidates" / "candidates.pt",
        "candidate_audit": root / "candidates" / "candidate_audit.json",
        "summaries": root / "selection" / "summaries.jsonl",
        "summary_placeholder": root / "selection" / "summaries_only_unused.pt",
        "scores": root / "selection" / "scores.jsonl",
        "score_manifest": root / "selection" / "score_manifest.json",
        "scored_summaries": root / "selection" / "scored_summaries.jsonl",
        "selected_indices": root / "selection" / "selected_indices.json",
        "selection_manifest": root / "selection" / "selection_manifest.json",
        "replay_shard": root / "replay" / "trace_shard.pt",
        "replay_summaries": root / "replay" / "summaries.jsonl",
        "replay_manifest": root / "replay" / "staged_manifest.json",
        "pipeline_manifest": root / "pipeline_manifest.json",
    }


def _python(config: PipelineConfig, tool: Path, *args: object) -> tuple[str, ...]:
    return (str(config.python), str(tool), *(str(value) for value in args))


def _summary_command(
    config: PipelineConfig, baseline: BaselineAdapter, paths: dict[str, Path]
) -> Command:
    protocol = _protocol(config)
    ratio = protocol["candidate"]["selection_ratio"]
    return Command(
        "summarize",
        _python(
            config,
            config.tools["summary_replay_builder"],
            "--candidate_dataset",
            paths["candidate_dataset"],
            "--target_dataset",
            baseline.source_dataset,
            "--protocol",
            config.protocol,
            "--refresh_cycle",
            config.refresh_cycle,
            "--policy_checkpoint",
            baseline.rollout_actor_checkpoint,
            "--output",
            paths["summary_placeholder"],
            "--selection",
            "scorer",
            "--select_ratio",
            ratio,
            "--selection_scope",
            "global",
            "--include_terminal_prefixes",
            "--allow_partial_candidate_set",
            "--reward_source",
            "dataset",
            "--summaries_only",
            "--summary_jsonl",
            paths["summaries"],
        ),
    )


def _eligibility_command(
    config: PipelineConfig, paths: dict[str, Path]
) -> Command:
    return Command(
        "summarize",
        _python(
            config,
            Path(__file__).resolve().parent / "annotate_eligibility.py",
            "--summaries",
            paths["summaries"],
            "--protocol",
            config.protocol,
            "--output",
            paths["summaries"],
        ),
    )


def _selection_commands(
    config: PipelineConfig, baseline: BaselineAdapter, paths: dict[str, Path]
) -> list[Command]:
    if config.selection_backend == "rule":
        scoring = [
            Command(
                "score",
                (
                    str(config.python),
                    "-m",
                    "trace_core.rule_score",
                    "--summaries",
                    paths["summaries"],
                    "--scores-output",
                    paths["scores"],
                    "--manifest-output",
                    paths["score_manifest"],
                    "--scored-summaries-output",
                    paths["scored_summaries"],
                ),
            )
        ]
    else:
        assert config.scorer_checkpoint is not None
        scoring = [
            Command(
                "score",
                (
                    str(config.python),
                    "-m",
                    "trace_scorer.score_candidates",
                    "--summaries",
                    paths["summaries"],
                    "--scorer",
                    config.scorer_checkpoint,
                    "--scores-output",
                    paths["scores"],
                    "--manifest-output",
                    paths["score_manifest"],
                ),
            ),
            Command(
                "select",
                (
                    str(config.python),
                    "-m",
                    "trace_scorer.attach_scores",
                    "--summaries",
                    paths["summaries"],
                    "--scores",
                    paths["scores"],
                    "--score-manifest",
                    paths["score_manifest"],
                    "--output",
                    paths["scored_summaries"],
                ),
            ),
        ]
    return [
        *scoring,
        Command(
            "select",
            _python(
                config,
                config.tools["distribution_selector"],
                "--summaries",
                paths["scored_summaries"],
                "--dataset",
                baseline.source_dataset,
                "--protocol",
                config.protocol,
                "--output",
                paths["selected_indices"],
                "--eligible-field",
                "trace_selection_eligible",
            ),
        ),
        Command(
            "select",
            (
                str(config.python),
                "-m",
                "trace_core.export_selection_manifest",
                "--summaries",
                paths["summaries"],
                "--scores",
                paths["scores"],
                "--score-manifest",
                paths["score_manifest"],
                "--candidate-dataset",
                paths["candidate_dataset"],
                "--selected-indices-json",
                paths["selected_indices"],
                "--output",
                paths["selection_manifest"],
            ),
        ),
    ]


def _replay_commands(
    config: PipelineConfig, baseline: BaselineAdapter, paths: dict[str, Path]
) -> list[Command]:
    protocol = _protocol(config)
    ratio = protocol["candidate"]["selection_ratio"]
    replay_environment = {
        "TRACE_REPLAY_ZERO_OBSERVATION_INDICES": json.dumps(
            list(baseline.replay_zero_observation_indices)
        ),
        "TRACE_REPLAY_REWARD_VERSION": baseline.replay_reward_version or "",
        "TRACE_REPLAY_RECOMPUTE_REWARD_AFTER_SELECTION": "1",
    }
    return [
        Command(
            "replay",
            _python(
                config,
                config.tools["summary_replay_builder"],
                "--candidate_dataset",
                paths["candidate_dataset"],
                "--target_dataset",
                baseline.source_dataset,
                "--protocol",
                config.protocol,
                "--refresh_cycle",
                config.refresh_cycle,
                "--policy_checkpoint",
                baseline.rollout_actor_checkpoint,
                "--output",
                paths["replay_shard"],
                "--selection",
                "scorer",
                "--select_ratio",
                ratio,
                "--selection_scope",
                "global",
                "--include_terminal_prefixes",
                "--allow_partial_candidate_set",
                "--selection_manifest",
                paths["selection_manifest"],
                "--reward_source",
                "rwm_aligned",
                "--reward_config_from_checkpoint",
                baseline.reward_config,
                "--summary_jsonl",
                paths["replay_summaries"],
            ),
            replay_environment,
        ),
        Command(
            "replay",
            _python(
                config,
                config.tools["replay_manifest"],
                "stage",
                "--output",
                paths["replay_manifest"],
                "--shard",
                paths["replay_shard"],
                "--cycle",
                config.refresh_cycle,
                "--protocol",
                config.protocol,
                "--side",
                baseline.setting,
                "--condition",
                baseline.name,
            ),
        ),
    ]


def _format_training(
    baseline: BaselineAdapter, config: PipelineConfig, paths: dict[str, Path]
) -> Command | None:
    if not baseline.training_command:
        return None
    values = {
        "repo_root": str(config.repo_root),
        "trace_manifest": str(paths["replay_manifest"]),
        "trace_replay_ratio": str(_protocol(config)["replay"]["ratio"]),
        "source_dataset": str(baseline.source_dataset),
        "rollout_actor_checkpoint": str(baseline.rollout_actor_checkpoint),
        "reward_config": str(baseline.reward_config),
        "world_model_checkpoint": str(baseline.world_model_checkpoint or ""),
        "training_initial_checkpoint": str(baseline.training_initial_checkpoint or ""),
        "output_root": str(config.output_root / "training"),
    }
    argv = tuple(item.format(**values) for item in baseline.training_command)
    environment = {
        key: value.format(**values)
        for key, value in baseline.training_environment.items()
    }
    environment.update(
        {
            "TRACE_REPLAY_PATH": values["trace_manifest"],
            "TRACE_REPLAY_RATIO": values["trace_replay_ratio"],
        }
    )
    return Command("train", argv, environment)


def _collect_commands(
    config: PipelineConfig, baseline: BaselineAdapter, paths: dict[str, Path]
) -> list[Command]:
    protocol = _protocol(config)
    candidate = protocol["candidate"]
    simulator = config.simulator
    branches = int(candidate["trajectories_per_start"])
    environments = int(simulator.get("environments_per_batch", 1024))
    if environments % branches:
        raise ValueError("simulator.environments_per_batch must divide by trajectories_per_start.")
    source_batch_size = environments // branches
    starts = int(candidate["source_start_count"])
    batches = (starts + source_batch_size - 1) // source_batch_size
    commands = [
        Command(
            "collect",
            _python(
                config,
                config.tools["source_selector"],
                "--dataset",
                baseline.source_dataset,
                "--output",
                paths["source_ids"],
                "--report",
                paths["source_report"],
                "--protocol",
                config.protocol,
                "--seed",
                int(simulator.get("source_selection_seed", 10421)),
                "--batch-output-dir",
                paths["source_batches"],
                "--batch-size",
                source_batch_size,
            ),
        )
    ]
    extra_args = [str(item) for item in simulator.get("extra_arguments", [])]
    wrapper = simulator.get("gpu_wrapper")
    for batch in range(batches):
        prefix: tuple[str, ...] = ()
        environment = None
        if wrapper:
            prefix = ("bash", str((config.repo_root / str(wrapper)).resolve()))
            environment = {"V10_GPU_POOL": str(simulator.get("gpu_pool", ""))}
        part = paths["candidate_parts"] / f"batch_{batch:02d}.pt"
        source_batch = paths["source_batches"] / f"source_ids_batch_{batch:02d}.pt"
        collector = _python(
            config,
            config.tools["candidate_collector"],
            "--task",
            str(simulator["task"]),
            "--device",
            str(simulator.get("device", "cuda:0")),
            "--seed",
            int(simulator.get("rollout_seed_base", 20430)) + batch,
            "--num_envs",
            environments,
            "--num_transitions",
            environments * int(candidate["horizon"]),
            "--save_path",
            part,
            "--expert_policy_path",
            baseline.rollout_actor_checkpoint,
            "--collector_mix",
            "expert:1.0",
            "--fixed_collector_assignment",
            "--collector_assignment_seed",
            42,
            "--trace_reset_dataset",
            baseline.source_dataset,
            "--trace_reset_mode",
            baseline.candidate_reset_mode,
            "--trace_source_ids_path",
            source_batch,
            "--trace_min_source_timestep",
            candidate["minimum_source_timestep"],
            "--trace_source_history_horizon",
            candidate["minimum_source_timestep"],
            "--trace_rollout_length",
            candidate["horizon"],
            "--trace_trajectories_per_state",
            branches,
            "--trace_action_temperature",
            candidate["action_temperature"],
            "--trace_identity_tolerance",
            protocol["validity"]["maximum_reset_reconstruction_error"],
            "--no-trace_require_source_transition_identity",
            "--headless",
            *extra_args,
        )
        commands.append(Command("collect", prefix + collector, environment))
    for batch in range(1, batches):
        base = (
            paths["candidate_parts"] / "batch_00.pt"
            if batch == 1
            else paths["candidate_parts"] / f"merged_through_{batch - 1:02d}.pt"
        )
        append = paths["candidate_parts"] / f"batch_{batch:02d}.pt"
        output = paths["candidate_parts"] / f"merged_through_{batch:02d}.pt"
        commands.append(
            Command(
                "collect",
                _python(
                    config,
                    config.tools["candidate_merger"],
                    "--base",
                    base,
                    "--append",
                    append,
                    "--output",
                    output,
                ),
            )
        )
    final_source = (
        paths["candidate_parts"] / f"merged_through_{batches - 1:02d}.pt"
        if batches > 1
        else paths["candidate_parts"] / "batch_00.pt"
    )
    commands.append(Command("collect", ("__copy__", str(final_source), str(paths["candidate_dataset"]))))
    commands.append(
        Command(
            "collect",
            _python(
                config,
                config.tools["candidate_auditor"],
                "--candidate",
                paths["candidate_dataset"],
                "--source-ids",
                paths["source_ids"],
                "--protocol",
                config.protocol,
                "--output",
                paths["candidate_audit"],
            ),
        )
    )
    return commands


def build_commands(
    config: PipelineConfig, baseline: BaselineAdapter
) -> tuple[dict[str, Path], list[Command]]:
    paths = _paths(config)
    commands = _collect_commands(config, baseline, paths)
    commands.append(_summary_command(config, baseline, paths))
    commands.append(_eligibility_command(config, paths))
    commands.extend(_selection_commands(config, baseline, paths))
    commands.extend(_replay_commands(config, baseline, paths))
    training = _format_training(baseline, config, paths)
    if training:
        commands.append(training)
    return paths, commands


def _preflight(config: PipelineConfig, baseline: BaselineAdapter) -> None:
    required = [
        config.python,
        config.protocol,
        baseline.source_dataset,
        baseline.rollout_actor_checkpoint,
        baseline.reward_config,
        *config.tools.values(),
    ]
    if config.scorer_checkpoint is not None:
        required.append(config.scorer_checkpoint)
    if baseline.world_model_checkpoint is not None:
        required.append(baseline.world_model_checkpoint)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"TRACE preflight missing inputs: {missing}")
    dependency_check = subprocess.run(
        [
            str(config.python),
            "-c",
            "import numpy, scipy, torch, yaml",
        ],
        cwd=config.repo_root,
        capture_output=True,
        text=True,
    )
    if dependency_check.returncode != 0:
        raise RuntimeError(
            "Configured TRACE Python cannot import numpy/scipy/torch/yaml: "
            f"{dependency_check.stderr.strip()}"
        )
    if baseline.training_initial_checkpoint is not None:
        actor = baseline.training_initial_checkpoint / "actor.pt"
        if not actor.is_file():
            raise FileNotFoundError(f"Training initial checkpoint has no actor.pt: {actor}")


def _write_manifest(
    config: PipelineConfig,
    baseline: BaselineAdapter,
    paths: dict[str, Path],
    commands: Iterable[Command],
) -> None:
    manifest = {
        "schema": "portable_trace_pipeline_run_v1",
        "baseline": {
            "name": baseline.name,
            "source_dataset": str(baseline.source_dataset),
            "source_dataset_sha256": _sha256(baseline.source_dataset),
            "rollout_actor_checkpoint": str(baseline.rollout_actor_checkpoint),
            "reward_config": str(baseline.reward_config),
            "reward_config_sha256": _sha256(baseline.reward_config),
            "training_initial_checkpoint": (
                str(baseline.training_initial_checkpoint)
                if baseline.training_initial_checkpoint
                else None
            ),
            "candidate_simulator": baseline.candidate_simulator,
            "candidate_reset_mode": baseline.candidate_reset_mode,
            "replay_zero_observation_indices": list(
                baseline.replay_zero_observation_indices
            ),
            "replay_reward_version": baseline.replay_reward_version,
            "replay_recompute_reward_after_selection": (
                baseline.replay_recompute_reward_after_selection
            ),
        },
        "trace": {
            "protocol": str(config.protocol),
            "protocol_sha256": _sha256(config.protocol),
            "selection_backend": config.selection_backend,
            "scorer_checkpoint": (
                str(config.scorer_checkpoint)
                if config.scorer_checkpoint is not None
                else None
            ),
            "scorer_checkpoint_sha256": (
                _sha256(config.scorer_checkpoint)
                if config.scorer_checkpoint is not None
                else None
            ),
            "simulator_is_upstream": True,
            "summary_reward_source": "ignored_by_scorer",
            "replay_reward_source": "baseline_adapter",
        },
        "commands": [
            {
                "stage": command.stage,
                "argv": [str(value) for value in command.argv],
                "environment": command.environment or {},
            }
            for command in commands
        ],
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["pipeline_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Comma-separated subset of: {','.join(STAGES)}",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    requested = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    unknown = sorted(set(requested) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown TRACE stages: {unknown}")
    config, baseline = load_configuration(args.pipeline, args.baseline)
    if not args.skip_preflight:
        _preflight(config, baseline)
    paths, commands = build_commands(config, baseline)
    _write_manifest(config, baseline, paths, commands)
    selected = [command for command in commands if command.stage in requested]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "portable_trace_pipeline_dry_run_v1",
                    "baseline": baseline.name,
                    "stages": requested,
                    "commands": [
                        {
                            "stage": row.stage,
                            "argv": [str(value) for value in row.argv],
                            "environment": row.environment or {},
                        }
                        for row in selected
                    ],
                    "pipeline_manifest": str(paths["pipeline_manifest"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    for path in paths.values():
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
    for command in selected:
        print(
            f"[TRACE:{command.stage}] "
            f"{' '.join(str(value) for value in command.argv)}",
            flush=True,
        )
        if command.argv[0] == "__copy__":
            shutil.copy2(command.argv[1], command.argv[2])
            continue
        environment = os.environ.copy()
        environment.update(command.environment or {})
        subprocess.run(
            tuple(str(value) for value in command.argv),
            cwd=config.repo_root,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
