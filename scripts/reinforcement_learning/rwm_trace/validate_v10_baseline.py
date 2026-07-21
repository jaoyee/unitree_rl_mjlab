"""Verify that a reusable no-TRACE baseline is exactly paired with V10 inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.v10_protocol import load_protocol, sha256_path


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--side", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--warmup", required=True)
    parser.add_argument("--reward-source", required=True)
    parser.add_argument("--training-steps", type=int, required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    _, _, protocol_sha = load_protocol(args.protocol)
    warmup = Path(args.warmup)
    expected = {
        "schema": "go2_trace_v10_baseline_manifest_v1",
        "protocol_sha256": protocol_sha,
        "side": args.side,
        "condition": args.condition,
        "dataset_sha256": sha256_path(args.dataset),
        "model_sha256": sha256_path(args.model),
        "warmup_actor_sha256": sha256_path(warmup / "actor.pt"),
        "warmup_replay_buffer_sha256": sha256_path(warmup / "replay_buffer.pt"),
        "seed": 300,
        "training_steps": args.training_steps,
        "reward_sha256": sha256_path(args.reward_source),
    }
    mismatches = {key: {"actual": data.get(key), "expected": value} for key, value in expected.items() if data.get(key) != value}
    policy = Path(str(data.get("policy", "")))
    if not policy.is_dir() or not (policy / "actor.pt").is_file() or data.get("policy_actor_sha256") != sha256_path(policy / "actor.pt"):
        mismatches["policy"] = "missing_or_changed"
    if mismatches:
        raise SystemExit(f"Baseline is not V10-compatible: {mismatches}")
    print(json.dumps({"passed": True, "policy": str(policy.resolve())}, indent=2))


if __name__ == "__main__":
    main()
