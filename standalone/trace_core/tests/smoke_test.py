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
            "schema": "trace_baseline_adapter_v1",
            "name": "from_zero_baseline",
            "setting": "sim",
            "source_dataset": "dataset.pt",
            "rollout_actor_checkpoint": "actor",
            "reward_config": "reward.yaml",
            "world_model_checkpoint": None,
            "candidate": {
                "simulator": "normal",
                "reset_mode": "exact_snapshot",
            },
            "replay": {
                "reward_version": "baseline_reward_v1",
                "recompute_reward_after_selection": True,
                "zero_observation_indices": [0, 1, 2],
            },
            "training": {"initial_checkpoint": None, "command": []},
        },
        root,
    )
    assert baseline.training_initial_checkpoint is None
    assert baseline.source_dataset == root / "dataset.pt"
    assert baseline.replay_zero_observation_indices == (0, 1, 2)

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
    collector = next(
        command for command in commands if "--trace_reset_mode" in command.argv
    )
    assert "exact_snapshot" in collector.argv
    assert str(baseline.rollout_actor_checkpoint) in collector.argv
    replay = next(
        command
        for command in commands
        if command.environment
        and "TRACE_REPLAY_ZERO_OBSERVATION_INDICES" in command.environment
    )
    assert replay.environment["TRACE_REPLAY_ZERO_OBSERVATION_INDICES"] == "[0, 1, 2]"
    assert replay.environment["TRACE_REPLAY_REWARD_VERSION"] == "baseline_reward_v1"
    print(json.dumps({"trace_core_smoke_test": "passed"}))


if __name__ == "__main__":
    main()
