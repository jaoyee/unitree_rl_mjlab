import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core.config import BaselineAdapter, PipelineConfig


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
            "training": {"initial_checkpoint": None, "command": []},
        },
        root,
    )
    assert baseline.training_initial_checkpoint is None
    assert baseline.source_dataset == root / "dataset.pt"

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
            "protocol": "protocol.json",
            "scorer_checkpoint": "scorer.pt",
            "output_root": "output",
            "simulator": {"task": "dummy"},
            "tools": tools,
        },
        root,
    )
    assert pipeline.repo_root == root
    assert set(pipeline.tools) == set(tools)
    print(json.dumps({"trace_core_smoke_test": "passed"}))


if __name__ == "__main__":
    main()
