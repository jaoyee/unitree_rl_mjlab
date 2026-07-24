from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core.config import BaselineAdapter


def v12_document() -> dict:
    return {
        "schema": "trace_baseline_adapter_v2",
        "name": "v12",
        "setting": "sim",
        "source_dataset": "dataset.pt",
        "rollout_actor_checkpoint": "rollout_actor",
        "reward_config": "rwm_flashsac_config.yaml",
        "world_model_checkpoint": "model_5000.pt",
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
    }


class V12ContractTest(unittest.TestCase):
    def test_current_v12_contract(self) -> None:
        adapter = BaselineAdapter.from_document(v12_document(), ROOT)
        self.assertEqual(adapter.actor_observation_dim, 45)
        self.assertEqual(adapter.critic_observation_dim, 48)
        self.assertEqual(adapter.replay_zero_observation_indices, ())
        self.assertTrue(adapter.source_leakage_audit_required)
        self.assertEqual(adapter.candidate_resample_groups, 5)
        self.assertEqual(
            adapter.candidate_rollout_policy_source,
            "baseline_actor_checkpoint",
        )

    def test_rejects_single_sided_blv_zeroing(self) -> None:
        document = v12_document()
        document["replay"]["zero_observation_indices"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "must be empty"):
            BaselineAdapter.from_document(document, ROOT)

    def test_rejects_policy_continuation(self) -> None:
        document = v12_document()
        document["training"]["policy_initialization"] = "resume"
        document["training"]["initial_checkpoint"] = "step4882"
        with self.assertRaisesRegex(ValueError, "initialize from zero"):
            BaselineAdapter.from_document(document, ROOT)

    def test_requires_source_leakage_audit(self) -> None:
        document = v12_document()
        document["replay"]["source_leakage_audit_required"] = False
        with self.assertRaisesRegex(ValueError, "source_leakage"):
            BaselineAdapter.from_document(document, ROOT)

    def test_rejects_expert_candidate_rollout(self) -> None:
        document = v12_document()
        document["candidate"]["rollout_policy_source"] = "expert_actor"
        with self.assertRaisesRegex(ValueError, "expert-policy"):
            BaselineAdapter.from_document(document, ROOT)

    def test_rejects_reward_drift(self) -> None:
        document = v12_document()
        document["replay"]["additional_reward_penalties"] = True
        with self.assertRaisesRegex(ValueError, "reward penalties"):
            BaselineAdapter.from_document(document, ROOT)

    def test_rejects_invalid_observation_indices(self) -> None:
        document = v12_document()
        document["observation"]["actor_excluded_indices"] = [0, 1, 48]
        with self.assertRaisesRegex(ValueError, "outside observation.full_dim"):
            BaselineAdapter.from_document(document, ROOT)

    def test_actor_must_exclude_unsupervised_outputs(self) -> None:
        document = v12_document()
        document["observation"]["actor_excluded_indices"] = [0, 1]
        document["observation"]["actor_dim"] = 46
        with self.assertRaisesRegex(ValueError, "excluded from the actor"):
            BaselineAdapter.from_document(document, ROOT)


if __name__ == "__main__":
    unittest.main()
