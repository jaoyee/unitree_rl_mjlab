from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.reinforcement_learning.rwm_trace.build_go2_partitioned_feedback_pairs import (
    _stable_partition,
)
from scripts.reinforcement_learning.rwm_trace.replay import REPLAY_KEYS, V10TraceReplaySampler
from scripts.reinforcement_learning.rwm_trace.select_v10_trajectories import (
    dataset_mode_bucket_probabilities,
    select_v10_trajectories,
)
from scripts.reinforcement_learning.rwm_trace.v10_protocol import (
    COMMAND_MODES,
    MODE_TO_ID,
    load_protocol,
    sha256_path,
    signed_magnitude_buckets,
)
from scripts.reinforcement_learning.rwm_trace.v10_metric_selection import (
    load_metric_config,
    select_global_metric_pilot,
)
from scripts.reinforcement_learning.rwm_trace.v10_replay_manifest import (
    commit_manifest,
    stage_manifest,
    verify_manifest,
)


COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "pure_x": (0.1, 0.0, 0.0),
    "pure_y": (0.0, 0.05, 0.0),
    "pure_yaw": (0.0, 0.0, 0.1),
    "xy": (0.1, 0.05, 0.0),
    "x_yaw": (0.1, 0.0, 0.1),
    "y_yaw": (0.0, 0.05, 0.1),
    "xy_yaw": (0.1, 0.05, 0.1),
}


class TraceV10ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol, self.protocol_path, self.protocol_sha = load_protocol()

    def test_frozen_protocol_values(self) -> None:
        candidate = self.protocol["candidate"]
        replay = self.protocol["replay"]
        training = self.protocol["training"]
        self.assertEqual(candidate["action_temperature"], 1.0)
        self.assertEqual(candidate["horizon"], 100)
        self.assertEqual(candidate["source_start_count"], 1024)
        self.assertEqual(candidate["trajectories_per_start"], 16)
        self.assertEqual(candidate["candidate_trajectory_count"], 16384)
        self.assertEqual(candidate["selected_trajectory_count"], 4096)
        self.assertEqual(sum(candidate["source_mode_counts"].values()), 1024)
        self.assertEqual(sum(candidate["selected_mode_counts"].values()), 4096)
        self.assertEqual(replay["trace_rows_per_batch"], 205)
        self.assertEqual(replay["retained_refreshes"], 5)
        self.assertEqual(training["refresh_cycles"], 8)
        self.assertEqual(training["environment_steps_per_refresh"], 5_000_000)
        self.assertFalse(replay["terminal_reward_override"])
        self.assertEqual(replay["failure_backprop_steps"], 0)
        self.assertEqual(replay["action_saturation_penalty_scale"], 0.0)
        self.assertEqual(replay["action_delta_penalty_scale"], 0.0)

    def test_old_protocol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            old = copy.deepcopy(self.protocol)
            old["protocol_version"] = "go2_trace_v9"
            path.write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(path)

    def test_global_selector_honors_exact_mode_quota(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["candidate"]["selected_trajectory_count"] = len(COMMAND_MODES)
        protocol["candidate"]["selected_mode_counts"] = {mode: 1 for mode in COMMAND_MODES}
        protocol["candidate"]["milp_time_limit_seconds"] = 30
        summaries: list[dict[str, float | str]] = []
        scores: list[float] = []
        for mode_index, mode in enumerate(COMMAND_MODES):
            command = COMMANDS[mode]
            for branch in range(2):
                summaries.append(
                    {
                        "command_mode": mode,
                        "command_vx_mean": command[0],
                        "command_vy_mean": command[1],
                        "command_yaw_mean": command[2],
                    }
                )
                scores.append(float(branch + mode_index * 0.01))
        targets = np.zeros(18, dtype=np.int64)
        for command in COMMANDS.values():
            for axis, bucket in enumerate(signed_magnitude_buckets(command, protocol)):
                if bucket >= 0:
                    targets[axis * 6 + bucket] += 1
        selected, diagnostics = select_v10_trajectories(
            summaries,
            np.asarray(scores),
            np.arange(len(summaries)),
            protocol=protocol,
            marginal_targets=targets,
        )
        self.assertEqual(len(selected), len(COMMAND_MODES))
        self.assertEqual(diagnostics["selected_mode_counts"], {mode: 1 for mode in COMMAND_MODES})
        self.assertTrue(all(scores[index] >= 1.0 for index in selected))

    def test_dataset_mode_bucket_probabilities_executes_all_signed_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.pt"
            commands = torch.as_tensor(list(COMMANDS.values()), dtype=torch.float32)
            torch.save({"commands": commands}, dataset_path)
            probabilities = np.asarray(
                dataset_mode_bucket_probabilities(dataset_path, self.protocol),
                dtype=np.float64,
            )
            self.assertEqual(probabilities.shape, (len(COMMAND_MODES), 3, 6))
            for mode_id, mode in enumerate(COMMAND_MODES):
                for axis, bucket in enumerate(signed_magnitude_buckets(COMMANDS[mode], self.protocol)):
                    total = float(probabilities[mode_id, axis].sum())
                    if bucket < 0:
                        self.assertEqual(total, 0.0)
                    else:
                        self.assertEqual(total, 1.0)
                        self.assertEqual(float(probabilities[mode_id, axis, bucket]), 1.0)

    def test_source_selector_emits_exact_four_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mode_names: list[str] = []
            for mode in COMMAND_MODES:
                mode_names.extend([mode] * int(self.protocol["candidate"]["source_mode_counts"][mode]))
            commands = torch.as_tensor([COMMANDS[mode] for mode in mode_names], dtype=torch.float32)
            time_steps = 33
            dataset = {
                "states": torch.zeros(time_steps, 1024, 45),
                "commands": commands.unsqueeze(0).repeat(time_steps, 1, 1),
                "episode_ids": torch.arange(1024).unsqueeze(0).repeat(time_steps, 1),
                "timesteps": torch.arange(time_steps).unsqueeze(1).repeat(1, 1024),
            }
            dataset_path = root / "dataset.pt"
            output = root / "source_ids.pt"
            report = root / "report.json"
            batches = root / "batches"
            torch.save(dataset, dataset_path)
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts/reinforcement_learning/rwm_trace/select_v10_source_starts.py"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--protocol",
                    str(self.protocol_path),
                    "--seed",
                    "42",
                    "--batch-output-dir",
                    str(batches),
                    "--batch-size",
                    "256",
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"source selector failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            artifact = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(len(artifact["source_ids"]), 1024)
            self.assertEqual(artifact["metadata"]["mode_counts"], self.protocol["candidate"]["source_mode_counts"])
            batch_paths = sorted(batches.glob("source_ids_batch_*.pt"))
            self.assertEqual(len(batch_paths), 4)
            self.assertTrue(
                all(
                    len(torch.load(path, map_location="cpu", weights_only=False)["source_ids"]) == 256
                    for path in batch_paths
                )
            )

    def test_partition_is_stable_across_refreshes(self) -> None:
        expected = _stable_partition(42, "x_yaw", "dataset-sha:source-17", 0.2)
        for _ in range(10):
            self.assertEqual(
                _stable_partition(42, "x_yaw", "dataset-sha:source-17", 0.2), expected
            )

    def test_metric_pilot_selects_exact_unconstrained_global_top25(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "scripts/reinforcement_learning/rwm_trace/trace_v10_metric_pilot.json"
        )
        config, _, _ = load_metric_config(config_path, base_protocol_sha256=self.protocol_sha)
        summaries = []
        for index in range(5000):
            quality = index / 4999.0
            summaries.append(
                {
                    "command_mode": "stand" if index < 4500 else "pure_x",
                    "command_vx_mean": 0.0 if index < 4500 else 0.2,
                    "command_vy_mean": 0.0,
                    "command_yaw_mean": 0.0,
                    "linear_tracking_error_steady": 1.0 - quality,
                    "yaw_tracking_error_steady": 1.0 - quality,
                    "tilt_mean": 1.0 - quality,
                    "tilt_max": 1.0 - quality,
                    "roll_pitch_rate_rms": 1.0 - quality,
                    "base_height_std": 1.0 - quality,
                    "survival_fraction": quality,
                }
            )
        selected, scores, diagnostics = select_global_metric_pilot(
            summaries,
            np.arange(len(summaries)),
            config=config,
            selection="metric",
            seed=42,
        )
        self.assertEqual(len(selected), 4096)
        self.assertFalse(diagnostics["command_distribution_constrained"])
        self.assertGreaterEqual(int(selected.min()), 904)
        self.assertTrue(np.isfinite(scores).all())

    def test_uniform_metric_sampler_does_not_restore_mode_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "trace_v10_protocol.json"
            protocol_path.write_bytes(self.protocol_path.read_bytes())
            self.protocol_sha = sha256_path(protocol_path)
            count = 100
            observation = torch.arange(count, dtype=torch.float32).reshape(count, 1).repeat(1, 48)
            shard = root / "uniform_shard.pt"
            torch.save(
                {
                    "format_version": "go2_trace_replay_v10_shard_v1",
                    "observation": observation,
                    "action": torch.zeros(count, 12),
                    "reward": torch.zeros(count, 1),
                    "terminated": torch.zeros(count, 1),
                    "truncated": torch.zeros(count, 1),
                    "next_observation": observation.clone(),
                    "command_mode_id": torch.zeros(count, dtype=torch.long),
                    "command_signed_buckets": torch.full((count, 3), -1, dtype=torch.int16),
                    "source_state_id": torch.arange(count),
                    "refresh_cycle": torch.ones(count, dtype=torch.long),
                    "metadata": {
                        "trace_protocol_version": "go2_trace_v10",
                        "protocol_sha256": self.protocol_sha,
                        "refresh_cycle": 1,
                        "policy_checkpoint": "incoming.pt",
                        "policy_checkpoint_sha256": "incoming-policy-sha",
                        "sampling_strategy": "uniform_selected_replay",
                        "selected_command_mode_counts": {"stand": count},
                        "target_mode_bucket_probabilities": np.zeros((8, 3, 6)).tolist(),
                    },
                },
                shard,
            )
            staged = root / "staged.json"
            stage_manifest(
                output=staged,
                committed_manifest=None,
                shard=shard,
                cycle=1,
                protocol_path=protocol_path,
                side="real",
                condition="rr05",
            )
            sampler = V10TraceReplaySampler(staged, seed=42)
            self.assertEqual(sampler.sampling_strategy, "uniform_selected_replay")
            batch = sampler.sample(205, device="cpu")
            self.assertEqual(batch["reward"].shape[0], 205)

    def test_all_v10_shell_python_heredocs_compile(self) -> None:
        root = Path(__file__).resolve().parents[1]
        shell_root = root / "scripts" / "reinforcement_learning" / "go2_sim_gap_aligned"
        scripts = sorted(shell_root.glob("*v10*.sh")) + [shell_root / "run_v10_gpu_stage.sh"]
        self.assertTrue(scripts)
        pattern = re.compile(r"<<'PY'\s*\n(.*?)\nPY\s*$", re.MULTILINE | re.DOTALL)
        compiled = 0
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            for index, snippet in enumerate(pattern.findall(source)):
                compile(snippet, f"{script.name}:heredoc:{index}", "exec")
                compiled += 1
        self.assertGreater(compiled, 0)

    def _write_shard(self, root: Path, cycle: int) -> Path:
        count = len(COMMAND_MODES)
        observation = torch.arange(count, dtype=torch.float32).reshape(count, 1).repeat(1, 48)
        actions = torch.zeros(count, 12)
        signed = torch.full((count, 3), -1, dtype=torch.int16)
        targets = np.zeros((count, 3, 6), dtype=np.float64)
        for mode_id, mode in enumerate(COMMAND_MODES):
            buckets = signed_magnitude_buckets(COMMANDS[mode], self.protocol)
            signed[mode_id] = torch.as_tensor(buckets, dtype=torch.int16)
            for axis, bucket in enumerate(buckets):
                if bucket >= 0:
                    targets[mode_id, axis, bucket] = 1.0
        payload = {
            "format_version": "go2_trace_replay_v10_shard_v1",
            "observation": observation,
            "action": actions,
            "reward": torch.zeros(count, 1),
            "terminated": torch.zeros(count, 1),
            "truncated": torch.zeros(count, 1),
            "next_observation": observation.clone(),
            "command_mode_id": torch.arange(count, dtype=torch.long),
            "command_signed_buckets": signed,
            "source_state_id": torch.arange(count, dtype=torch.long),
            "refresh_cycle": torch.full((count,), cycle, dtype=torch.long),
            "metadata": {
                "trace_protocol_version": "go2_trace_v10",
                "protocol_sha256": self.protocol_sha,
                "refresh_cycle": cycle,
                "policy_checkpoint": "incoming.pt",
                "policy_checkpoint_sha256": "incoming-policy-sha",
                "scorer_checkpoint_sha256": "scorer-sha",
                "selected_command_mode_counts": {mode: 1 for mode in COMMAND_MODES},
                "target_mode_bucket_probabilities": targets.tolist(),
            },
        }
        path = root / f"cycle_{cycle:02d}" / "replay.pt"
        path.parent.mkdir(parents=True)
        torch.save(payload, path)
        return path

    def test_last_five_manifest_and_exact_sampler_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "trace_v10_protocol.json"
            protocol_path.write_bytes(self.protocol_path.read_bytes())
            # The shard must reference the SHA of the copied immutable protocol.
            self.protocol_sha = sha256_path(protocol_path)
            manifest = root / "replay_manifest.json"
            expired: list[Path] = []
            for cycle in range(1, 8):
                shard = self._write_shard(root, cycle)
                if cycle <= 2:
                    expired.append(shard)
                staged = root / f"staged_{cycle}.json"
                stage_manifest(
                    output=staged,
                    committed_manifest=manifest if manifest.is_file() else None,
                    shard=shard,
                    cycle=cycle,
                    protocol_path=protocol_path,
                    side="real",
                    condition="rr05",
                )
                policy = root / f"policy_{cycle}.pt"
                policy.write_bytes(f"policy-{cycle}".encode())
                commit_manifest(
                    staged=staged,
                    output=manifest,
                    policy_checkpoint=policy,
                    delete_expired=True,
                )
            verified = verify_manifest(manifest, allow_pending=False)
            self.assertEqual([row["cycle"] for row in verified["shards"]], [3, 4, 5, 6, 7])
            self.assertTrue(all(not path.exists() for path in expired))
            sampler = V10TraceReplaySampler(manifest, seed=7)
            batch = sampler.sample(205, device="cpu")
            self.assertEqual(set(batch), set(REPLAY_KEYS))
            self.assertEqual(batch["reward"].shape[0], 205)
            sampled_modes = batch["observation"][:, 0].to(torch.long)
            expected_counts = sampler._mode_counts(205)
            # Reset the carry because the first call above already advanced it.
            sampler.mode_carry[:] = 0.0
            expected_counts = sampler._mode_counts(205)
            actual_counts = torch.bincount(sampled_modes, minlength=len(COMMAND_MODES)).numpy()
            np.testing.assert_array_equal(actual_counts, expected_counts)


if __name__ == "__main__":
    unittest.main()
