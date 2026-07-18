#!/usr/bin/env python3
"""Strictly merge controlled TRACE replay artifacts from multiple Go2 gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path


REPLAY_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", action="append", required=True, metavar="GAP=PATH")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _parse_inputs(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected GAP=PATH, got {item!r}.")
        gap, raw_path = item.split("=", 1)
        if not gap or gap in seen:
            raise ValueError(f"Empty or duplicate gap identifier: {gap!r}.")
        seen.add(gap)
        parsed.append((gap, Path(raw_path).expanduser().resolve()))
    return parsed


def _signature(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
        metadata.get("trace_protocol_version"),
        int(metadata.get("n_step", -1)),
        float(metadata.get("gamma", float("nan"))),
        metadata.get("reward_source"),
        metadata.get("selection"),
        float(metadata.get("select_ratio", float("nan"))),
        int(metadata.get("trajectory_length", -1)),
    )


def main() -> None:
    args = _parse_args()
    inputs = _parse_inputs(args.input)
    payloads: list[tuple[str, Path, dict[str, Any]]] = []
    reference_signature: tuple[Any, ...] | None = None
    for gap, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("trace_protocol_version") != "go2_trace_v5_controlled":
            raise ValueError(f"{gap}: replay is not controlled V5.")
        missing = [key for key in REPLAY_KEYS if not isinstance(payload.get(key), torch.Tensor)]
        if missing:
            raise ValueError(f"{gap}: missing tensor fields {missing}.")
        count = int(payload["observation"].shape[0])
        if any(int(payload[key].shape[0]) != count for key in REPLAY_KEYS):
            raise ValueError(f"{gap}: replay fields have inconsistent row counts.")
        signature = _signature(metadata)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"{gap}: replay semantics differ: {signature} != {reference_signature}.")
        payloads.append((gap, path, payload))

    merged = {
        key: torch.cat([payload[key].detach().cpu() for _, _, payload in payloads], dim=0)
        for key in REPLAY_KEYS
    }
    first_metadata = dict(payloads[0][2]["metadata"])
    source_rows = {
        gap: int(payload["observation"].shape[0]) for gap, _, payload in payloads
    }
    total = sum(source_rows.values())
    merged["format_version"] = "go2_trace_replay_v2"
    merged["metadata"] = {
        **first_metadata,
        "trace_protocol_version": "go2_trace_v5_controlled",
        "merge_schema": "go2_trace_multigap_replay_v5",
        "transition_count": total,
        "source_rows": source_rows,
        "source_fractions": {gap: rows / total for gap, rows in source_rows.items()},
        "source_paths": {gap: str(path) for gap, path, _ in payloads},
        "source_sha256": {gap: sha256_path(path) for gap, path, _ in payloads},
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(json.dumps({
        "output": str(output),
        "output_sha256": sha256_path(output),
        "transition_count": total,
        "source_rows": source_rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
