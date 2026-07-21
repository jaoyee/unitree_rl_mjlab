"""Atomic last-five replay manifests for Go2 TRACE V10."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.v10_protocol import load_protocol


MANIFEST_SCHEMA = "go2_trace_v10_replay_manifest_v1"
SHARD_FORMAT = "go2_trace_replay_v10_shard_v1"


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported V10 replay manifest: {path}")
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _shard_row(path: Path, cycle: int, protocol_sha: str, *, status: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format_version") != SHARD_FORMAT:
        raise ValueError(f"Not a TRACE V10 replay shard: {path}")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("trace_protocol_version") != "go2_trace_v10":
        raise ValueError(f"Shard has the wrong TRACE protocol: {path}")
    if metadata.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"Shard protocol SHA differs from the active V10 protocol: {path}")
    if int(metadata.get("refresh_cycle", -1)) != int(cycle):
        raise ValueError(f"Shard refresh cycle does not match {cycle}: {path}")
    return {
        "cycle": int(cycle),
        "path": str(path.resolve()),
        "sha256": sha256_path(path),
        "transition_count": int(payload["reward"].shape[0]),
        "status": status,
        "policy_checkpoint": metadata.get("policy_checkpoint"),
        "policy_checkpoint_sha256": metadata.get("policy_checkpoint_sha256"),
        "scorer_checkpoint_sha256": metadata.get("scorer_checkpoint_sha256"),
        "sampling_strategy": metadata.get(
            "sampling_strategy", "dataset_mode_signed_magnitude_stratified"
        ),
        "selected_mode_counts": metadata.get("selected_command_mode_counts"),
        "signed_marginal_counts": (
            (metadata.get("selection_diagnostics") or {}).get("selected_signed_marginal_counts")
        ),
    }


def stage_manifest(
    *,
    output: Path,
    committed_manifest: Path | None,
    shard: Path,
    cycle: int,
    protocol_path: Path,
    side: str,
    condition: str,
) -> dict[str, Any]:
    protocol, resolved_protocol, protocol_sha = load_protocol(protocol_path)
    retention = int(protocol["replay"]["retained_refreshes"])
    previous: list[dict[str, Any]] = []
    if committed_manifest is not None and committed_manifest.is_file():
        old = _read(committed_manifest)
        if old.get("protocol_sha256") != protocol_sha:
            raise ValueError("Committed replay manifest uses another protocol SHA.")
        if old.get("side") != side or old.get("condition") != condition:
            raise ValueError("Committed replay manifest belongs to another experiment condition.")
        previous = [row for row in old.get("shards", []) if row.get("status") == "committed"]
    previous = [row for row in previous if int(row["cycle"]) < int(cycle)]
    previous = sorted(previous, key=lambda row: int(row["cycle"]))[-(retention - 1) :]
    current = _shard_row(shard.resolve(), cycle, protocol_sha, status="pending")
    strategies = {row.get("sampling_strategy") for row in [*previous, current]}
    if len(strategies) != 1:
        raise ValueError("Retained V10 replay shards cannot mix sampling strategies.")
    data = {
        "schema": MANIFEST_SCHEMA,
        "protocol_version": "go2_trace_v10",
        "protocol_path": str(resolved_protocol),
        "protocol_sha256": protocol_sha,
        "side": side,
        "condition": condition,
        "retained_refreshes": retention,
        "active_cycle": int(cycle),
        "sampling_strategy": current["sampling_strategy"],
        "shards": [*previous, current],
    }
    _atomic_write(output, data)
    return data


def commit_manifest(
    *, staged: Path, output: Path, policy_checkpoint: Path, delete_expired: bool = False
) -> dict[str, Any]:
    previous = _read(output) if output.is_file() else None
    data = _read(staged)
    policy_sha = sha256_path(policy_checkpoint)
    pending = [row for row in data["shards"] if row.get("status") == "pending"]
    if len(pending) != 1 or int(pending[0]["cycle"]) != int(data["active_cycle"]):
        raise ValueError("A staged V10 manifest must contain exactly one current pending shard.")
    if pending[0].get("policy_checkpoint_sha256") == policy_sha:
        raise ValueError(
            "The shard provenance points to the incoming policy; commit requires a newly trained policy hash."
        )
    for row in data["shards"]:
        path = Path(row["path"])
        if not path.is_file() or sha256_path(path) != row["sha256"]:
            raise ValueError(f"Replay shard changed before commit: {path}")
        row["status"] = "committed"
    data["committed_policy_checkpoint"] = str(policy_checkpoint.resolve())
    data["committed_policy_checkpoint_sha256"] = policy_sha
    data["status"] = "committed"
    _atomic_write(output, data)
    if delete_expired and previous is not None:
        retained = {Path(row["path"]).resolve() for row in data["shards"]}
        run_root = output.parent.resolve()
        for row in previous.get("shards", []):
            path = Path(row["path"]).resolve()
            if path not in retained and path.is_file() and run_root in path.parents:
                path.unlink()
    return data


def verify_manifest(path: Path, *, allow_pending: bool) -> dict[str, Any]:
    data = _read(path)
    _, _, protocol_sha = load_protocol(data["protocol_path"])
    if protocol_sha != data.get("protocol_sha256"):
        raise ValueError("Replay manifest protocol SHA is stale.")
    if len(data.get("shards", [])) > int(data["retained_refreshes"]):
        raise ValueError("Replay manifest exceeds its retention horizon.")
    for row in data["shards"]:
        if row.get("status") == "pending" and not allow_pending:
            raise ValueError("Committed replay manifest contains a pending shard.")
        shard = Path(row["path"])
        if not shard.is_file() or sha256_path(shard) != row["sha256"]:
            raise ValueError(f"Replay shard is missing or changed: {shard}")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--output", required=True)
    stage.add_argument("--committed-manifest")
    stage.add_argument("--shard", required=True)
    stage.add_argument("--cycle", type=int, required=True)
    stage.add_argument("--protocol", required=True)
    stage.add_argument("--side", choices=("sim", "real"), required=True)
    stage.add_argument("--condition", required=True)
    commit = subparsers.add_parser("commit")
    commit.add_argument("--staged", required=True)
    commit.add_argument("--output", required=True)
    commit.add_argument("--policy-checkpoint", required=True)
    commit.add_argument("--delete-expired", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--allow-pending", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "stage":
        data = stage_manifest(
            output=Path(args.output),
            committed_manifest=Path(args.committed_manifest) if args.committed_manifest else None,
            shard=Path(args.shard),
            cycle=args.cycle,
            protocol_path=Path(args.protocol),
            side=args.side,
            condition=args.condition,
        )
    elif args.command == "commit":
        data = commit_manifest(
            staged=Path(args.staged),
            output=Path(args.output),
            policy_checkpoint=Path(args.policy_checkpoint),
            delete_expired=bool(args.delete_expired),
        )
    else:
        data = verify_manifest(Path(args.manifest), allow_pending=bool(args.allow_pending))
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
