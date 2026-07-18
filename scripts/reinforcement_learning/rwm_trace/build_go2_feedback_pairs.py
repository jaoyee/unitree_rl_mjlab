#!/usr/bin/env python3
"""Build active Go2 trajectory pairs and strict-JSON LLM prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.scorer import load_scorer_checkpoint, score_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer_checkpoint", default=None)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument(
        "--pair_prefix",
        default="go2_pair",
        help="Stable prefix used to keep pair IDs unique across TRACE refresh cycles.",
    )
    parser.add_argument(
        "--cross_start_fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of cross-start comparisons. Controlled Go2 TRACE defaults "
            "to zero because different commands/states are not counterfactual branches."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prompt(row: dict) -> str:
    source_kinds = {
        str(row[side].get("source_kind", "simulator_rollout"))
        for side in ("trajectory_i", "trajectory_j")
    }
    source_description = (
        "two contiguous expert-dataset windows used to pretrain the TRACE scorer"
        if source_kinds == {"expert_dataset_window"}
        else "two short Go2 trajectories produced by an imperfect simulator"
    )
    payload = {
        "pair_id": row["pair_id"],
        "instruction": (
            f"Compare {source_description}. Select the trajectory that is "
            "more useful and less artifact-dominated as replay for locomotion learning. Prefer stable upright motion, "
            "coherent command tracking, smooth non-saturated actions, plausible state changes, sustained motion, and "
            "reasonable contact switching. Simulator return is only one diagnostic and must not decide the label by "
            "itself. If evidence is insufficient or contradictory, use tie; use invalid for malformed/nonfinite data."
        ),
        "same_start_state": row["same_start_state"],
        "trajectory_i": row["trajectory_i"],
        "trajectory_j": row["trajectory_j"],
        "required_json": {
            "pair_id": row["pair_id"],
            "feedback": "i | j | tie | invalid",
            "confidence": "number in [0, 1]",
            "reason": "short reason",
        },
    }
    return "Return strict JSON only.\n" + json.dumps(payload, indent=2, sort_keys=True)


def _start_key(summary: dict) -> str:
    return str(summary.get("start_state_key", summary.get("start_state_id", -1)))


def _comparison_group_key(summary: dict) -> str:
    return str(summary.get("comparison_group_key", _start_key(summary)))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.cross_start_fraction <= 1.0:
        raise ValueError("--cross_start_fraction must be in [0, 1].")
    summaries = _read_jsonl(args.summaries)
    if len(summaries) < 2:
        raise ValueError("Need at least two trajectory summaries.")
    rng = np.random.default_rng(args.seed)
    if args.scorer_checkpoint:
        model, stats, _ = load_scorer_checkpoint(args.scorer_checkpoint)
        scores = score_summaries(model, summaries, stats)
    else:
        scores = np.zeros(len(summaries), dtype=np.float32)

    by_start: dict[str, list[int]] = {}
    for index, summary in enumerate(summaries):
        by_start.setdefault(_comparison_group_key(summary), []).append(index)
    # (score_gap, left, right, within_controlled_group, exact_same_start)
    candidates: list[tuple[float, int, int, bool, bool]] = []
    for indices in by_start.values():
        order = sorted(indices, key=lambda index: float(scores[index]))
        for left, right in zip(order[0::2], order[1::2]):
            same_exact_start = _start_key(summaries[left]) == _start_key(summaries[right])
            candidates.append((abs(float(scores[left] - scores[right])), left, right, True, same_exact_start))
    cross_count = int(round(args.budget * args.cross_start_fraction))
    all_indices = np.arange(len(summaries))
    rng.shuffle(all_indices)
    for left, right in zip(all_indices[0::2], all_indices[1::2]):
        if _start_key(summaries[left]) == _start_key(summaries[right]):
            continue
        candidates.append((abs(float(scores[left] - scores[right])), int(left), int(right), False, False))

    same = sorted((item for item in candidates if item[3]), key=lambda item: item[0])
    cross = sorted((item for item in candidates if not item[3]), key=lambda item: item[0])[:cross_count]
    selected = same[: max(0, args.budget - len(cross))] + cross
    rng.shuffle(selected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for pair_index, (gap, left, right, _within_group, same_start) in enumerate(selected):
            row = {
                "pair_id": f"{args.pair_prefix}_{pair_index:06d}",
                "pair_score_gap": gap,
                "same_start_state": same_start,
                "trajectory_i": summaries[left],
                "trajectory_j": summaries[right],
            }
            row["prompt"] = _prompt(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(selected)} Go2 feedback pairs to {output}")


if __name__ == "__main__":
    main()
