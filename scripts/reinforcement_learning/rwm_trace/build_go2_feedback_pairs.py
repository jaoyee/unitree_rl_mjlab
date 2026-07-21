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

from scripts.reinforcement_learning.rwm_trace.reference_summary import (
    attach_references_to_summaries,
    load_reference_map,
)
from scripts.reinforcement_learning.rwm_trace.scorer import load_scorer_checkpoint, score_summaries


CONDITION_PROFILES = {
    "general": {
        "name": "generic locomotion scorer pretraining",
        "primary_objective": "overall dynamic locomotion quality",
        "ranking_rule": (
            "Prefer upright, command-coherent, sustained motion with plausible contacts and smooth actions."
        ),
    },
    "g0": {
        "name": "healthy-control complement",
        "primary_objective": "dynamic command tracking without sacrificing balance",
        "ranking_rule": (
            "Prefer trajectories that improve command tracking and dynamic balance relative to the matched RWM "
            "reference. Do not reward merely standing still or duplicating an already-easy stationary behavior "
            "when commanded motion is present."
        ),
    },
    "p5": {
        "name": "5 kg payload complement",
        "primary_objective": "recover velocity tracking from payload-affected states",
        "ranking_rule": (
            "Give first priority to simulator candidates that improve linear and yaw tracking relative to the "
            "matched payload RWM reference while keeping upright balance. A stationary-looking trajectory is not "
            "useful merely because it is stable."
        ),
    },
    "p75": {
        "name": "7.5 kg payload complement",
        "primary_objective": "recover velocity tracking under the stronger payload gap while preserving dynamic balance",
        "ranking_rule": (
            "Prefer meaningful tracking improvement relative to the matched payload RWM reference, but reject gains "
            "that create large tilt, implausible contacts, or impending failure. Balance is a stricter constraint than p5."
        ),
    },
    "rr05": {
        "name": "RR-calf KP/KD x0.5 complement",
        "primary_objective": "dynamic balance and rear-right support during commanded motion",
        "ranking_rule": (
            "Prefer candidates that improve rear-right support, gait continuity, bounded tilt, and commanded motion "
            "relative to the matched RR-calf RWM reference. Do not reward standing still simply because it is easy."
        ),
    },
    "rr03": {
        "name": "RR-calf KP/KD x0.3 complement",
        "primary_objective": "prevent rear-right collapse and preserve dynamic balance under severe actuator weakening",
        "ranking_rule": (
            "This gap is more severe than rr05. Prioritize candidates that avoid rear-right kneeling/collapse, "
            "limit roll and tilt accumulation, maintain plausible support/contact transitions, and sustain a "
            "continuous gait relative to the matched RR-calf RWM reference. Use velocity tracking next."
        ),
    },
}


DATASET_WINDOW_RANKING_RULES = {
    "general": (
        "Prefer the expert window with better upright, command-coherent, sustained locomotion, plausible contacts, "
        "and smooth non-saturated actions."
    ),
    "g0": (
        "Prefer the expert window that best demonstrates healthy dynamic command tracking and balance. Do not reward "
        "standing still when commanded motion is present."
    ),
    "p5": (
        "Prefer the expert window that best demonstrates payload-robust velocity tracking while staying upright. "
        "A stable but motionless window is not useful unless the command is stand."
    ),
    "p75": (
        "Prefer the expert window that best demonstrates stronger-payload robustness: bounded tilt, plausible support, "
        "and meaningful command tracking. Reject windows that look stable only because motion collapsed."
    ),
    "rr05": (
        "Prefer the expert window with better rear-right support, gait continuity, bounded tilt, and commanded motion "
        "under the RR-calf weakened condition."
    ),
    "rr03": (
        "Prefer the expert window that avoids rear-right kneeling/collapse, limits roll/tilt accumulation, maintains "
        "plausible support/contact transitions, and still attempts commanded motion under severe RR-calf weakening."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer_checkpoint", default=None)
    parser.add_argument(
        "--reference_summaries",
        default=None,
        help="JSONL summaries of same-start frozen-RWM rollouts used to build residual scorer features.",
    )
    parser.add_argument(
        "--condition",
        choices=tuple(CONDITION_PROFILES),
        default="general",
        help="Human-introduced gap whose missing capability this scorer must target.",
    )
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
    parser.add_argument(
        "--same_command_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only compare trajectories from the same command mode.",
    )
    parser.add_argument(
        "--balance_command_modes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allocate the pair budget across available command modes before filling leftovers.",
    )
    parser.add_argument(
        "--required_pair_modes",
        nargs="*",
        default=[],
        help="Command modes that must be represented when --min_pairs_per_command_mode is positive.",
    )
    parser.add_argument(
        "--min_pairs_per_command_mode",
        type=int,
        default=0,
        help="Minimum selected pair count per required command mode; generates same-mode cross-group pairs if needed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prompt(row: dict, condition: str) -> str:
    source_kinds = {
        str(row[side].get("source_kind", "simulator_rollout"))
        for side in ("trajectory_i", "trajectory_j")
    }
    is_dataset_window_pair = source_kinds == {"expert_dataset_window"}
    source_description = (
        "two contiguous expert-dataset windows used to pretrain the TRACE scorer"
        if is_dataset_window_pair
        else "two short Go2 trajectories produced by an imperfect simulator"
    )
    profile = CONDITION_PROFILES[condition]
    if is_dataset_window_pair:
        comparison_semantics = (
            "trajectory_i and trajectory_j are real contiguous windows from the condition-specific expert dataset. "
            "They are used only to pretrain a condition-aware TRACE preference scorer before active simulator "
            "feedback begins. These expert windows do not have matched RWM references, and no candidate-vs-RWM delta "
            "fields should be expected or required."
        )
        instruction = (
            f"Compare {source_description}. Select the window that is a better positive example for the stated "
            "condition's desired behavior. Apply the condition_profile priorities, but judge only the provided expert "
            "dataset-window metrics. For every non-stand command, explicitly inspect projected net displacement, "
            "steady-state linear/yaw tracking, direction-correct fraction, foot swing/contact transitions, survival, "
            "tilt, and action smoothness/saturation. Stable in-place twisting or feet remaining planted under a "
            f"motion command is not successful locomotion. {DATASET_WINDOW_RANKING_RULES[condition]} "
            "Do not use absent RWM reference or delta information. Expert-window return is only one diagnostic and "
            "must not decide the label by itself. Prefer plausible, non-artifact-dominated transitions. If evidence "
            "is insufficient or contradictory, use tie; use invalid for malformed/nonfinite data."
        )
    else:
        comparison_semantics = (
            "trajectory_i and trajectory_j are nominal-imperfect-simulator candidates. Each candidate may include "
            "a matched rwm_reference_summary generated from the same source dataset state, same command, same frozen "
            "RWM, and current policy. Delta fields use the convention candidate minus RWM for returns/survival/contact "
            "and RWM minus candidate for errors/tilt/action costs, so positive delta usually means the simulator "
            "candidate complements an RWM weakness."
        )
        instruction = (
            f"Compare {source_description}. Select the trajectory that is "
            "more useful as complementary replay for capabilities that the matched gap-conditioned RWM does not "
            "already provide. The source reset belongs to the stated condition; the rollout simulator itself is "
            "nominal, so nominal payload_mass=0 or rr_calf_strength=1 metadata describes the imperfect simulator "
            "and must not be used as the task identity. Apply the condition_profile priorities and the candidate-vs-RWM "
            f"delta fields. {profile['ranking_rule']} For every non-stand command, explicitly inspect projected net "
            "displacement, steady-state tracking, direction-correct fraction, yaw displacement, foot swing counts, "
            "longest continuous stance, and foot clearance. Stable body twisting, feet glued to the ground, or "
            "in-place rotation under a translation command is not successful locomotion. A high rear-foot contact "
            "fraction alone is not evidence of healthy support. Simulator return is only one diagnostic and must not decide "
            "the label by itself. Prefer plausible, non-artifact-dominated transitions. If evidence is insufficient "
            "or contradictory, use tie; use invalid for malformed/nonfinite data."
        )
    payload = {
        "pair_id": row["pair_id"],
        "condition": condition,
        "condition_profile": profile,
        "comparison_semantics": comparison_semantics,
        "instruction": instruction,
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


def _command_mode(summary: dict) -> str:
    return str(summary.get("command_mode", "unknown"))


def _pair_key(left: int, right: int) -> tuple[int, int]:
    a, b = sorted((int(left), int(right)))
    return a, b


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.cross_start_fraction <= 1.0:
        raise ValueError("--cross_start_fraction must be in [0, 1].")
    summaries = _read_jsonl(args.summaries)
    references = load_reference_map(args.reference_summaries) if args.reference_summaries else None
    summaries = attach_references_to_summaries(
        summaries,
        references,
        strict=bool(args.reference_summaries),
    )
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
        mode_groups: dict[str, list[int]] = {}
        if args.same_command_mode:
            for index in indices:
                mode_groups.setdefault(str(summaries[index].get("command_mode", "unknown")), []).append(index)
        else:
            mode_groups["all"] = indices
        for mode_indices in mode_groups.values():
            order = sorted(mode_indices, key=lambda index: float(scores[index]))
            for left, right in zip(order[0::2], order[1::2]):
                same_exact_start = _start_key(summaries[left]) == _start_key(summaries[right])
                candidates.append((abs(float(scores[left] - scores[right])), left, right, True, same_exact_start))
    cross_count = int(round(args.budget * args.cross_start_fraction))
    all_indices = np.arange(len(summaries))
    rng.shuffle(all_indices)
    for left, right in zip(all_indices[0::2], all_indices[1::2]):
        if _start_key(summaries[left]) == _start_key(summaries[right]):
            continue
        if args.same_command_mode and summaries[left].get("command_mode") != summaries[right].get("command_mode"):
            continue
        candidates.append((abs(float(scores[left] - scores[right])), int(left), int(right), False, False))

    if args.min_pairs_per_command_mode > 0:
        required_modes = tuple(args.required_pair_modes) or tuple(
            sorted({_command_mode(summary) for summary in summaries})
        )
        seen = {_pair_key(item[1], item[2]) for item in candidates}
        by_mode_indices: dict[str, list[int]] = {}
        for index, summary in enumerate(summaries):
            by_mode_indices.setdefault(_command_mode(summary), []).append(index)
        for mode in required_modes:
            existing = sum(
                1
                for item in candidates
                if item[3]
                and _command_mode(summaries[item[1]]) == mode
                and _command_mode(summaries[item[2]]) == mode
            )
            need = max(0, int(args.min_pairs_per_command_mode) - existing)
            if need <= 0:
                continue
            indices = list(by_mode_indices.get(mode, ()))
            if len(indices) < 2:
                raise ValueError(
                    f"Cannot form required command-mode pairs for {mode}: only {len(indices)} summaries."
                )
            rng.shuffle(indices)
            max_bucket_count = min(4, len(indices) // 2)

            def bucket_capacity(bucket: list[int]) -> int:
                return len(bucket) * (len(bucket) - 1) // 2

            bucket_count = 1
            for candidate_bucket_count in range(max_bucket_count, 1, -1):
                candidate_buckets = [
                    list(indices[offset::candidate_bucket_count])
                    for offset in range(candidate_bucket_count)
                ]
                if all(len(bucket) >= 2 for bucket in candidate_buckets) and (
                    sum(bucket_capacity(bucket) for bucket in candidate_buckets) >= need
                ):
                    bucket_count = candidate_bucket_count
                    break
            buckets = [list(indices[offset::bucket_count]) for offset in range(bucket_count)]
            bucket_pools: list[list[tuple[float, int, int, bool, bool]]] = []
            for bucket in buckets:
                pool: list[tuple[float, int, int, bool, bool]] = []
                for pos, left in enumerate(bucket):
                    for right in bucket[pos + 1:]:
                        key = _pair_key(left, right)
                        if key in seen:
                            continue
                        same_exact_start = _start_key(summaries[left]) == _start_key(summaries[right])
                        gap = abs(float(scores[left] - scores[right]))
                        pool.append((gap, int(left), int(right), True, same_exact_start))
                        seen.add(key)
                pool.sort(key=lambda item: item[0])
                if pool:
                    bucket_pools.append(pool)
            if sum(len(pool) for pool in bucket_pools) < need:
                raise ValueError(
                    f"Cannot satisfy min_pairs_per_command_mode for {mode}: "
                    f"existing={existing}, extra={sum(len(pool) for pool in bucket_pools)}, "
                    f"required={args.min_pairs_per_command_mode}."
                )
            extra: list[tuple[float, int, int, bool, bool]] = []
            while len(extra) < need:
                progressed = False
                for pool in bucket_pools:
                    if pool and len(extra) < need:
                        extra.append(pool.pop(0))
                        progressed = True
                if not progressed:
                    break
            candidates.extend(extra[:need])

    same = sorted((item for item in candidates if item[3]), key=lambda item: item[0])
    cross = sorted((item for item in candidates if not item[3]), key=lambda item: item[0])[:cross_count]
    same_budget = max(0, args.budget - len(cross))

    def component_balanced_take(
        pool: list[tuple[float, int, int, bool, bool]],
        limit: int,
        used_keys: set[tuple[int, int]],
    ) -> list[tuple[float, int, int, bool, bool]]:
        available = [item for item in pool if _pair_key(item[1], item[2]) not in used_keys]
        if not available or limit <= 0:
            return []
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for item in available:
            union(_comparison_group_key(summaries[item[1]]), _comparison_group_key(summaries[item[2]]))
        by_component: dict[str, list[tuple[float, int, int, bool, bool]]] = {}
        for item in available:
            root = find(_comparison_group_key(summaries[item[1]]))
            by_component.setdefault(root, []).append(item)
        buckets = [sorted(items, key=lambda item: item[0]) for _, items in sorted(by_component.items())]
        selected_items: list[tuple[float, int, int, bool, bool]] = []
        while len(selected_items) < limit:
            progressed = False
            for bucket in buckets:
                while bucket and _pair_key(bucket[0][1], bucket[0][2]) in used_keys:
                    bucket.pop(0)
                if bucket and len(selected_items) < limit:
                    item = bucket.pop(0)
                    selected_items.append(item)
                    used_keys.add(_pair_key(item[1], item[2]))
                    progressed = True
                if len(selected_items) >= limit:
                    break
            if not progressed:
                break
        return selected_items

    if args.balance_command_modes and same_budget:
        by_mode: dict[str, list[tuple[float, int, int, bool, bool]]] = {}
        for item in same:
            by_mode.setdefault(_command_mode(summaries[item[1]]), []).append(item)
        selected_same = []
        used: set[tuple[int, int]] = set()
        required_modes = tuple(args.required_pair_modes) or tuple(sorted(by_mode))
        min_count = max(0, int(args.min_pairs_per_command_mode))
        if min_count > 0:
            required_total = min_count * len(required_modes)
            if required_total > same_budget:
                raise ValueError(
                    f"Pair budget {same_budget} is smaller than required command-mode minimum "
                    f"{min_count} x {len(required_modes)} = {required_total}."
                )
            for mode in required_modes:
                pool = by_mode.get(mode, [])
                if len(pool) < min_count:
                    raise ValueError(
                        f"Cannot reserve required pairs for {mode}: have {len(pool)}, need {min_count}."
                    )
                reserved = component_balanced_take(pool, min_count, used)
                if len(reserved) < min_count:
                    raise ValueError(
                        f"Cannot reserve leak-splittable required pairs for {mode}: "
                        f"selected={len(reserved)}, need={min_count}."
                    )
                selected_same.extend(reserved)
        remaining_budget = max(0, same_budget - len(selected_same))
        if remaining_budget:
            target = max(1, remaining_budget // max(1, len(by_mode)))
            for mode in sorted(by_mode):
                added = 0
                for item in by_mode[mode]:
                    key = _pair_key(item[1], item[2])
                    if key in used:
                        continue
                    selected_same.append(item)
                    used.add(key)
                    added += 1
                    if added >= target or len(selected_same) >= same_budget:
                        break
        if len(selected_same) < same_budget:
            remaining = [item for item in same if _pair_key(item[1], item[2]) not in used]
            selected_same.extend(remaining[: same_budget - len(selected_same)])
    else:
        selected_same = same[:same_budget]
    selected = selected_same + cross
    if args.min_pairs_per_command_mode > 0:
        required_modes = tuple(args.required_pair_modes) or tuple(
            sorted({_command_mode(summary) for summary in summaries})
        )
        selected_mode_counts = {mode: 0 for mode in required_modes}
        for _gap, left, right, *_ in selected:
            left_mode = _command_mode(summaries[left])
            right_mode = _command_mode(summaries[right])
            if left_mode == right_mode and left_mode in selected_mode_counts:
                selected_mode_counts[left_mode] += 1
        missing = {
            mode: count
            for mode, count in selected_mode_counts.items()
            if count < int(args.min_pairs_per_command_mode)
        }
        if missing:
            raise ValueError(
                "Selected feedback pairs do not satisfy command-mode minimums: "
                f"required={args.min_pairs_per_command_mode}, counts={selected_mode_counts}"
            )
    rng.shuffle(selected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for pair_index, (gap, left, right, _within_group, same_start) in enumerate(selected):
            if bool(rng.integers(0, 2)):
                left, right = right, left
            row = {
                "pair_id": f"{args.pair_prefix}_{pair_index:06d}",
                "pair_score_gap": gap,
                "same_start_state": same_start,
                "trajectory_i": summaries[left],
                "trajectory_j": summaries[right],
                "condition": args.condition,
            }
            row["prompt"] = _prompt(row, args.condition)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(selected)} Go2 feedback pairs to {output}")


if __name__ == "__main__":
    main()
