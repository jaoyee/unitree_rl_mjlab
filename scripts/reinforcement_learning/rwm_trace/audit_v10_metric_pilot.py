#!/usr/bin/env python3
"""Fail-closed audit for scorer-free V10 metric/random replay shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.v10_metric_selection import load_metric_config
from scripts.reinforcement_learning.rwm_trace.v10_protocol import load_protocol


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--metric-config", required=True)
    parser.add_argument("--selection", choices=("metric", "random"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol, _, protocol_sha = load_protocol(args.protocol)
    config, config_path, config_sha = load_metric_config(
        args.metric_config, base_protocol_sha256=protocol_sha
    )
    replay = torch.load(args.replay, map_location="cpu", weights_only=False, mmap=True)
    metadata = dict(replay.get("metadata") or {})
    summaries = [
        json.loads(line)
        for line in Path(args.summaries).read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [row for row in summaries if bool(row.get("selected"))]
    expected_candidates = int(protocol["candidate"]["candidate_trajectory_count"])
    expected_selected = int(config["selection"]["selected_trajectory_count"])
    errors: list[str] = []
    if replay.get("format_version") != "go2_trace_replay_v10_shard_v1":
        errors.append("bad replay format")
    if not expected_selected <= len(summaries) <= expected_candidates:
        errors.append(
            f"summary_count={len(summaries)} outside [{expected_selected}, {expected_candidates}]"
        )
    if len(selected) != expected_selected:
        errors.append(f"selected_count={len(selected)} expected={expected_selected}")
    if metadata.get("selection") != args.selection:
        errors.append("selection metadata mismatch")
    if metadata.get("sampling_strategy") != "uniform_selected_replay":
        errors.append("metric/random replay is not configured for uniform sampling")
    if metadata.get("metric_pilot_config_sha256") != config_sha:
        errors.append("metric config SHA mismatch")
    if metadata.get("scorer_checkpoint") is not None:
        errors.append("scorer checkpoint must be absent")
    if any(not bool(row.get("command_motion_gate_passed")) for row in selected):
        errors.append("selected replay contains a validity-rejected trajectory")
    scores = np.asarray([float(row.get("trace_score", np.nan)) for row in selected])
    if not np.isfinite(scores).all():
        errors.append("selected scores are non-finite")
    mode_counts = Counter(str(row.get("command_mode")) for row in selected)
    metadata_counts = {str(k): int(v) for k, v in metadata.get("selected_command_mode_counts", {}).items()}
    if dict(sorted(mode_counts.items())) != dict(sorted(metadata_counts.items())):
        errors.append("selected command-mode report disagrees with replay metadata")
    report = {
        "schema": "go2_trace_v10_metric_pilot_audit_v1",
        "passed": not errors,
        "selection": args.selection,
        "protocol_sha256": protocol_sha,
        "metric_config": str(config_path),
        "metric_config_sha256": config_sha,
        "replay_sha256": sha256_path(args.replay),
        "candidate_count": len(summaries),
        "candidate_rollout_attempt_count": expected_candidates,
        "selected_count": len(selected),
        "selected_fraction_of_rollout_attempts": float(len(selected) / expected_candidates),
        "selected_fraction_of_summarized_candidates": float(
            len(selected) / max(len(summaries), 1)
        ),
        "selected_mode_counts": dict(sorted(mode_counts.items())),
        "selected_signed_bucket_counts": metadata.get("selected_signed_bucket_counts"),
        "selected_score_mean": float(scores.mean()) if len(scores) else None,
        "selected_score_min": float(scores.min()) if len(scores) else None,
        "selected_score_max": float(scores.max()) if len(scores) else None,
        "errors": errors,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
