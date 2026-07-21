from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for package, relative in (
    ("src", "src"),
    ("src.tasks", "src/tasks"),
    ("src.tasks.rwm_velocity", "src/tasks/rwm_velocity"),
    ("src.tasks.rwm_velocity.mdp", "src/tasks/rwm_velocity/mdp"),
):
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(REPO_ROOT / relative)]
        sys.modules[package] = module


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_module("src.tasks.rwm_velocity.mdp.extractors", "src/tasks/rwm_velocity/mdp/extractors.py")
_rewards = _load_module("src.tasks.rwm_velocity.mdp.rewards", "src/tasks/rwm_velocity/mdp/rewards.py")
Go2RWMRewardState = _rewards.Go2RWMRewardState
compute_go2_imagination_reward = _rewards.compute_go2_imagination_reward


def reward_for(command: tuple[float, float, float], velocity: tuple[float, float, float], yaw: float = 0.0) -> float:
    state = torch.zeros(1, 45)
    state[:, 0:3] = torch.tensor(velocity)
    state[:, 5] = yaw
    state[:, 8] = -1.0
    action = torch.zeros(1, 12)
    reward_state = Go2RWMRewardState.create(1, 12, "cpu", 0.02, reward_version="v2_hierarchical")
    reward, _ = compute_go2_imagination_reward(
        state=state,
        action=action,
        command=torch.tensor([command]),
        foot_contact=torch.ones(1, 4),
        episode_length=torch.zeros(1, dtype=torch.long),
        reward_state=reward_state,
        epistemic_uncertainty=torch.zeros(1),
    )
    return float(reward.item())


class Go2HierarchicalRewardTest(unittest.TestCase):
    def test_reward_v2_pilot_is_fresh_and_protocol_bound(self) -> None:
        config_path = (
            REPO_ROOT
            / "scripts/reinforcement_learning/rwm_trace/trace_v10_reward_v2_pilot.json"
        )
        protocol_path = (
            REPO_ROOT / "scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        self.assertEqual(config["base_protocol_sha256"], protocol_sha)
        self.assertEqual(config["training"]["policy_resume_mode"], "actor_only")
        self.assertFalse(config["training"]["load_replay_buffer"])
        self.assertFalse(config["training"]["load_reward_normalizer"])
        self.assertEqual(config["reward_v2"]["version"], "v2_hierarchical")

        launcher = (
            REPO_ROOT
            / "scripts/reinforcement_learning/go2_sim_gap_aligned/launch_go2_reward_v2_rr05_pilot.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("POLICY_RESUME_MODE=actor_only", launcher)
        self.assertIn("LOAD_REPLAY_BUFFER=false", launcher)
        self.assertIn("LOAD_REWARD_NORMALIZER=false", launcher)
        snippets = re.findall(r"<<'PY'\s*\n(.*?)\nPY\s*$", launcher, re.MULTILINE | re.DOTALL)
        self.assertTrue(snippets)
        for index, snippet in enumerate(snippets):
            compile(snippet, f"reward_v2_launcher:heredoc:{index}", "exec")

    def test_v1_formula_is_unchanged(self) -> None:
        state = torch.zeros(1, 45)
        state[:, 8] = -1.0
        reward_state = Go2RWMRewardState.create(1, 12, "cpu", 0.02, reward_version="v1")
        reward, _ = compute_go2_imagination_reward(
            state,
            torch.zeros(1, 12),
            torch.tensor([[0.3, 0.0, 0.0]]),
            torch.ones(1, 4),
            torch.zeros(1, dtype=torch.long),
            reward_state,
            torch.zeros(1),
        )
        expected = (5.0 * torch.exp(torch.tensor(-0.09 / 0.25)) + 3.0 + 0.4) * 0.02
        self.assertAlmostEqual(float(reward.item()), float(expected.item()), places=7)

    def test_active_command_prefers_correct_motion_to_static_and_reverse(self) -> None:
        command = (0.3, 0.0, 0.0)
        correct = reward_for(command, (0.3, 0.0, 0.0))
        static = reward_for(command, (0.0, 0.0, 0.0))
        reverse = reward_for(command, (-0.3, 0.0, 0.0))
        self.assertGreater(correct, 0.05)
        self.assertLess(static, 0.0)
        self.assertGreater(correct, static + 0.03)
        self.assertGreater(static, reverse)

    def test_active_command_rejects_rotation_in_place_as_linear_response(self) -> None:
        command = (0.3, 0.0, 0.0)
        correct = reward_for(command, (0.3, 0.0, 0.0), yaw=0.0)
        rotate_in_place = reward_for(command, (0.0, 0.0, 0.0), yaw=0.3)
        self.assertGreater(correct, rotate_in_place + 0.03)

    def test_compound_command_requires_both_active_components(self) -> None:
        command = (0.3, 0.0, 0.3)
        complete = reward_for(command, (0.3, 0.0, 0.0), yaw=0.3)
        linear_only = reward_for(command, (0.3, 0.0, 0.0), yaw=0.0)
        yaw_only = reward_for(command, (0.0, 0.0, 0.0), yaw=0.3)
        self.assertGreater(complete, 0.08)
        self.assertGreater(complete, linear_only + 0.02)
        self.assertGreater(complete, yaw_only + 0.08)

    def test_stand_uses_stand_objective(self) -> None:
        stable = reward_for((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), yaw=0.0)
        drifting = reward_for((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), yaw=0.3)
        self.assertGreater(stable, drifting)

    def test_unknown_reward_version_fails_closed(self) -> None:
        state = torch.zeros(1, 45)
        reward_state = Go2RWMRewardState.create(1, 12, "cpu", 0.02, reward_version="unknown")
        with self.assertRaisesRegex(ValueError, "Unknown Go2 reward version"):
            compute_go2_imagination_reward(
                state,
                torch.zeros(1, 12),
                torch.tensor([[0.3, 0.0, 0.0]]),
                torch.ones(1, 4),
                torch.zeros(1, dtype=torch.long),
                reward_state,
                torch.zeros(1),
            )


if __name__ == "__main__":
    unittest.main()
