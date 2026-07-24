import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core.config import BaselineAdapter, PipelineConfig
from trace_core.run_pipeline import build_commands


def main() -> None:
    root = Path(tempfile.mkdtemp()).resolve()
    baseline = BaselineAdapter.from_document(
        {
            "schema": "trace_baseline_adapter_v2",
            "name": "v12_from_zero_baseline",
            "setting": "sim",
            "source_dataset": "dataset.pt",
            "rollout_actor_checkpoint": "actor",
            "reward_config": "reward.yaml",
            "world_model_checkpoint": None,
            "candidate": {
                "simulator": "normal",
                "reset_mode": "exact_snapshot",
                "resample_groups": 5,
                "rollout_policy_source": "baseline_actor_checkpoint",
            },
            "observation": {
                "full_dim": 48,
                "actor_dim": 45,
                "critic_dim": 48,
                "actor_excluded_indices": [0, 1, 2],
                "unsupervised_world_model_output_indices": [0, 1, 2],
            },
            "replay": {
                "reward_version": "v1_1_dense_progress",
                "recompute_reward_after_selection": True,
                "n_step": 3,
                "gamma": 0.99,
                "terminal_reward_override": False,
                "observation_policy": "source_native_full_state",
                "zero_observation_indices": [],
                "source_leakage_audit_required": True,
                "epistemic_uncertainty_mode": "physical_zero",
                "additional_reward_penalties": False,
            },
            "training": {
                "policy_initialization": "from_zero",
                "initial_checkpoint": None,
                "command": [],
            },
        },
        root,
    )
    assert baseline.training_initial_checkpoint is None
    assert baseline.source_dataset == root / "dataset.pt"
    assert baseline.replay_zero_observation_indices == ()
    assert baseline.actor_excluded_observation_indices == (0, 1, 2)
    assert baseline.training_policy_initialization == "from_zero"

    tools = {
        name: f"adapters/{name}.py"
        for name in (
            "source_selector",
            "candidate_collector",
            "candidate_merger",
            "candidate_auditor",
            "summary_replay_builder",
            "distribution_selector",
            "replay_manifest",
        )
    }
    pipeline = PipelineConfig.from_document(
        {
            "schema": "portable_trace_pipeline_v1",
            "repo_root": str(root),
            "python": "venv/python",
            "protocol": str(ROOT / "protocol.json"),
            "scorer_checkpoint": "scorer.pt",
            "output_root": "output",
            "simulator": {"task": "dummy"},
            "tools": tools,
        },
        root,
    )
    assert pipeline.repo_root == root
    assert set(pipeline.tools) == set(tools)
    rule_pipeline = PipelineConfig.from_document(
        {
            "schema": "portable_trace_pipeline_v1",
            "repo_root": str(root),
            "python": "venv/python",
            "protocol": str(ROOT / "protocol.json"),
            "selection": {"backend": "rule", "scorer_checkpoint": None},
            "output_root": "rule-output",
            "candidate_group_index": 3,
            "simulator": {"task": "dummy"},
            "tools": tools,
        },
        root,
    )
    assert rule_pipeline.selection_backend == "rule"
    assert rule_pipeline.scorer_checkpoint is None
    _, commands = build_commands(rule_pipeline, baseline)
    flat_commands = [
        " ".join(str(value) for value in command.argv) for command in commands
    ]
    assert any("trace_core.rule_score" in command for command in flat_commands)
    assert not any("trace_scorer." in command for command in flat_commands)
    assert not any("expert" in command.lower() for command in flat_commands)
    collector = next(
        command for command in commands if "--trace_reset_mode" in command.argv
    )
    assert "exact_snapshot" in collector.argv
    assert str(baseline.rollout_actor_checkpoint) in collector.argv
    assert collector.argv[collector.argv.index("--seed") + 1] == "320430"
    replay = next(
        command
        for command in commands
        if command.environment
        and "TRACE_REPLAY_ZERO_OBSERVATION_INDICES" in command.environment
    )
    assert replay.environment["TRACE_REPLAY_ZERO_OBSERVATION_INDICES"] == "[]"
    assert replay.environment["TRACE_REPLAY_REWARD_VERSION"] == "v1_1_dense_progress"
    assert replay.environment["TRACE_REPLAY_OBSERVATION_POLICY"] == (
        "source_native_full_state"
    )
    print(json.dumps({"trace_core_smoke_test": "passed"}))


if __name__ == "__main__":
    main()
