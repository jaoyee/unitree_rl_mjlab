"""Create or verify an atomic technical commit for one TRACE V10 refresh."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.v10_protocol import load_protocol, sha256_path
from scripts.reinforcement_learning.rwm_trace.v10_replay_manifest import verify_manifest


def _payload(policy: Path, scorer: Path, manifest: Path, protocol_path: Path, cycle: int) -> dict:
    _, resolved_protocol, protocol_sha = load_protocol(protocol_path)
    actor = policy / "actor.pt"
    buffer = policy / "replay_buffer.pt"
    if not actor.is_file() or not buffer.is_file():
        raise ValueError(f"V10 policy checkpoint is incomplete: {policy}")
    data = verify_manifest(manifest, allow_pending=False)
    if int(data.get("active_cycle", -1)) != cycle or data.get("protocol_sha256") != protocol_sha:
        raise ValueError("Committed replay manifest does not belong to this V10 refresh.")
    replay_buffers = list(policy.parent.glob("step*/replay_buffer.pt"))
    if replay_buffers != [buffer]:
        raise ValueError(
            "V10 policy output must contain one final replay buffer and no intermediate copies: "
            f"{replay_buffers}"
        )
    return {
        "schema": "go2_trace_v10_cycle_commit_v1",
        "cycle": cycle,
        "protocol_path": str(resolved_protocol),
        "protocol_sha256": protocol_sha,
        "policy_path": str(policy.resolve()),
        "policy_actor_sha256": sha256_path(actor),
        "policy_replay_buffer_sha256": sha256_path(buffer),
        "scorer_path": str(scorer.resolve()),
        "scorer_sha256": sha256_path(scorer),
        "replay_manifest_path": str(manifest.resolve()),
        "replay_manifest_sha256": sha256_path(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--policy", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = _payload(
        Path(args.policy).expanduser().resolve(),
        Path(args.scorer).expanduser().resolve(),
        Path(args.manifest).expanduser().resolve(),
        Path(args.protocol),
        args.cycle,
    )
    output = Path(args.output).expanduser().resolve()
    if args.command == "verify":
        actual = json.loads(output.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("V10 cycle commit is stale or belongs to different artifacts.")
        print(json.dumps(actual, indent=2, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
