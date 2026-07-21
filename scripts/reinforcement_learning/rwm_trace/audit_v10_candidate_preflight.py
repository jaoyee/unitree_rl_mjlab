"""Candidate-only V10 saturation, quota, and per-mode action-support audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_trace.v10_protocol import (
    COMMAND_MODES,
    MODE_TO_ID,
    classify_command,
    load_protocol,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol, _, protocol_sha = load_protocol(args.protocol)
    preflight = protocol["preflight"]
    replay = torch.load(args.replay, map_location="cpu", weights_only=False, mmap=True)
    metadata = dict(replay.get("metadata") or {})
    failures: list[str] = []
    if metadata.get("protocol_sha256") != protocol_sha:
        failures.append("protocol_sha")
    expected_modes = protocol["candidate"]["selected_mode_counts"]
    if metadata.get("selected_command_mode_counts") != expected_modes:
        failures.append("selected_mode_counts")

    summaries = [row for row in _read_jsonl(Path(args.summaries)) if bool(row.get("selected"))]
    saturation = np.asarray(
        [float(row.get("action_saturation_fraction", float("nan"))) for row in summaries],
        dtype=np.float64,
    )
    mean_any = float(np.mean(saturation)) if len(saturation) else float("nan")
    p95_any = float(np.quantile(saturation, 0.95)) if len(saturation) else float("nan")
    actions = torch.as_tensor(replay["action"]).reshape(-1, 12).float()
    element_saturation = float((actions.abs() >= float(protocol["validity"]["action_saturation_threshold"])).float().mean())
    if not np.isfinite(mean_any) or mean_any > float(preflight["maximum_selected_mean_any_joint_saturation"]):
        failures.append("mean_any_joint_saturation")
    if not np.isfinite(p95_any) or p95_any > float(preflight["maximum_selected_p95_any_joint_saturation"]):
        failures.append("p95_any_joint_saturation")
    if element_saturation > float(preflight["maximum_selected_element_saturation"]):
        failures.append("element_saturation")

    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False, mmap=True)
    dataset_actions = stack_time_key(dataset, "actions").reshape(-1, 12).float().numpy()
    dataset_commands = stack_time_key(dataset, "commands").reshape(-1, 3).float().numpy()
    dataset_mode_ids = np.asarray([MODE_TO_ID[classify_command(command)] for command in dataset_commands])
    replay_modes = torch.as_tensor(replay["command_mode_id"]).reshape(-1).numpy()
    replay_actions = actions.numpy()
    mode_ood: dict[str, float] = {}
    for mode in COMMAND_MODES:
        mode_id = MODE_TO_ID[mode]
        reference = dataset_actions[dataset_mode_ids == mode_id]
        values = replay_actions[replay_modes == mode_id]
        if len(reference) < 2 or len(values) == 0:
            mode_ood[mode] = float("nan")
            failures.append(f"action_ood_support:{mode}")
            continue
        low = np.quantile(reference, 0.01, axis=0)
        high = np.quantile(reference, 0.99, axis=0)
        fraction = float(np.mean((values < low) | (values > high)))
        mode_ood[mode] = fraction
        if fraction > float(preflight["maximum_mode_action_ood_fraction"]):
            failures.append(f"action_ood:{mode}")
    report = {
        "schema": "go2_trace_v10_candidate_preflight_v1",
        "selected_trajectory_count": len(summaries),
        "selected_mean_any_joint_saturation": mean_any,
        "selected_p95_any_joint_saturation": p95_any,
        "selected_action_element_saturation": element_saturation,
        "per_mode_action_ood_fraction": mode_ood,
        "failures": failures,
        "passed": not failures,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"V10 candidate-only preflight failed: {failures}")


if __name__ == "__main__":
    main()
