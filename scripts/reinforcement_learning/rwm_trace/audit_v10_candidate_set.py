"""Verify one fixed-batch V10 candidate collection before scorer work begins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.v10_protocol import COMMAND_MODES, classify_command, load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-ids", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol, _, protocol_sha = load_protocol(args.protocol)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False, mmap=True)
    source = torch.load(args.source_ids, map_location="cpu", weights_only=False)
    expected_ids = torch.as_tensor(source["source_ids"] if isinstance(source, dict) else source).long().reshape(-1)
    start_ids = torch.as_tensor(candidate.get("trace_start_state_ids")).long().reshape(-1)
    branches = int(protocol["candidate"]["trajectories_per_start"])
    errors: list[str] = []
    if len(start_ids) != int(protocol["candidate"]["candidate_trajectory_count"]):
        errors.append("candidate_count")
    unique, counts = torch.unique(start_ids, return_counts=True)
    if len(unique) != len(expected_ids) or set(unique.tolist()) != set(expected_ids.tolist()):
        errors.append("source_ids")
    if len(counts) and not bool((counts == branches).all()):
        errors.append("branches_per_source")
    commands = torch.stack(candidate["commands"], dim=0)[0]
    mode_counts = {mode: 0 for mode in COMMAND_MODES}
    for source_id in expected_ids.tolist():
        rows = torch.nonzero(start_ids == source_id, as_tuple=False).flatten()
        if len(rows) != branches:
            continue
        mode_counts[classify_command(commands[int(rows[0])].tolist())] += 1
    if mode_counts != protocol["candidate"]["source_mode_counts"]:
        errors.append("source_mode_counts")
    trace = dict((candidate.get("metadata") or {}).get("trace_candidates") or {})
    checks = {
        "action_temperature": trace.get("action_temperature") == protocol["candidate"]["action_temperature"],
        "horizon": trace.get("rollout_length") == protocol["candidate"]["horizon"],
        "branches": trace.get("trajectories_per_state") == branches,
        "valid_mask": candidate.get("trace_valid_masks") is not None,
        "one_step_identity": bool((trace.get("one_step_identity") or {}).get("passed", False)),
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    report = {
        "schema": "go2_trace_v10_candidate_audit_v1",
        "protocol_sha256": protocol_sha,
        "candidate_count": len(start_ids),
        "unique_source_count": len(unique),
        "source_mode_counts": mode_counts,
        "checks": checks,
        "errors": errors,
        "passed": not errors,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(f"V10 candidate audit failed: {errors}")


if __name__ == "__main__":
    main()
