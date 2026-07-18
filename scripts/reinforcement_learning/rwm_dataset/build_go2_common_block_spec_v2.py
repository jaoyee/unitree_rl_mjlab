"""Build a shared 25K command-block specification from multiple Go2 datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


MODE_TARGETS = {
    "stand": 2000, "pure_x": 6250, "pure_y": 2500, "pure_yaw": 2000,
    "xy": 3500, "x_yaw": 4250, "y_yaw": 1250, "xy_yaw": 3250,
}
MODE = {
    (False, False, False): "stand", (True, False, False): "pure_x",
    (False, True, False): "pure_y", (False, False, True): "pure_yaw",
    (True, True, False): "xy", (True, False, True): "x_yaw",
    (False, True, True): "y_yaw", (True, True, True): "xy_yaw",
}
AXES = ("x", "y", "yaw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument("--expert-collector-id", type=int, default=1)
    parser.add_argument("--zero-epsilon", type=float, default=1e-3)
    parser.add_argument("--x-edges", type=float, nargs=4, default=(0.05, 0.20, 0.35, 0.50))
    parser.add_argument("--y-edges", type=float, nargs=4, default=(0.03, 0.087, 0.143, 0.20))
    parser.add_argument("--yaw-edges", type=float, nargs=4, default=(0.05, 0.167, 0.283, 0.40))
    parser.add_argument("--minimum-block", type=int, default=40)
    return parser.parse_args()


def stack(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.stack(value)


def time_env(value: Any, feature: int | None = None) -> torch.Tensor:
    tensor = stack(value)
    if feature is not None:
        if tensor.ndim == 2:
            tensor = tensor[:, None, :]
        return tensor
    while tensor.ndim > 2 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor[:, None] if tensor.ndim == 1 else tensor


def axis_bin(value: torch.Tensor, edges: tuple[float, ...], epsilon: float) -> str | None:
    number = float(value)
    magnitude = abs(number)
    if magnitude <= epsilon:
        return "zero"
    if magnitude < edges[0] or magnitude > edges[-1] + 1e-6:
        return None
    bucket = min(2, sum(magnitude >= edge for edge in edges[1:-1]))
    return ("pos" if number > 0 else "neg") + f"_b{bucket}"


def stratum(command: torch.Tensor, edges: tuple[tuple[float, ...], ...], epsilon: float) -> str | None:
    bins = tuple(axis_bin(command[i], edges[i], epsilon) for i in range(3))
    if any(item is None for item in bins):
        return None
    active = tuple(item != "zero" for item in bins)
    return "|".join((MODE[active], *bins))


def capacities(path: Path, args: argparse.Namespace) -> tuple[dict[str, list[int]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    commands = time_env(data["commands"], 3)
    episodes = time_env(data["episode_ids"])
    timesteps = time_env(data["timesteps"])
    terminations = time_env(data["terminations"])
    allowed = torch.ones(commands.shape[:2], dtype=torch.bool)
    if args.expert_only:
        allowed &= time_env(data["collector_types"]) == int(args.expert_collector_id)
    allowed &= terminations <= 0.5
    edges = (tuple(args.x_edges), tuple(args.y_edges), tuple(args.yaw_edges))
    result: dict[str, list[int]] = defaultdict(list)
    for env in range(commands.shape[1]):
        start = 0
        current: str | None = None
        for t in range(commands.shape[0] + 1):
            label = stratum(commands[t, env], edges, args.zero_epsilon) if t < commands.shape[0] and allowed[t, env] else None
            contiguous = (
                t > start and label == current
                and int(episodes[t, env]) == int(episodes[t - 1, env])
                and int(timesteps[t, env]) == int(timesteps[t - 1, env]) + 1
            ) if t < commands.shape[0] else False
            if t == start:
                current = label
                continue
            if contiguous:
                continue
            if current is not None and t - start >= args.minimum_block:
                result[current].append(t - start)
            start, current = t, label
    summary = {
        key: {
            "segments": len(lengths),
            "steps": sum(lengths),
            "blocks": sum(length // args.minimum_block for length in lengths),
            "max_segment": max(lengths),
        }
        for key, lengths in sorted(result.items())
    }
    return dict(result), summary


def choose_blocks(
    all_capacities: dict[str, dict[str, list[int]]], minimum_block: int
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    shared = set.intersection(*(set(rows) for rows in all_capacities.values()))
    common_units = {
        key: min(
            sum(length // minimum_block for length in rows[key])
            for rows in all_capacities.values()
        )
        for key in shared
    }
    output: dict[str, list[int]] = defaultdict(list)
    diagnostics: dict[str, Any] = {}
    for mode, target in MODE_TARGETS.items():
        candidates = sorted(key for key in shared if key.split("|", 1)[0] == mode and common_units[key] > 0)
        if not candidates:
            raise ValueError(f"mode {mode} has no shared >= {minimum_block}-step stratum")
        block_count, remainder = divmod(target, minimum_block)
        lengths = [minimum_block] * block_count
        if remainder:
            lengths[-1] += remainder
        remaining = {key: common_units[key] for key in candidates}
        marginal = {axis: defaultdict(int) for axis in AXES}
        allocations = defaultdict(int)
        for length in sorted(lengths, reverse=True):
            needed_units = (length + minimum_block - 1) // minimum_block
            feasible = [
                key for key in candidates
                if remaining[key] >= needed_units
                and all(max(rows[key]) >= length for rows in all_capacities.values())
            ]
            if not feasible:
                raise ValueError(f"mode {mode} lacks shared block capacity for {target} steps")
            def score(key: str) -> tuple[float, float, str]:
                bins = key.split("|")[1:]
                imbalance = sum(marginal[axis][bucket] for axis, bucket in zip(AXES, bins, strict=True) if bucket != "zero")
                utilization = allocations[key] / max(1, common_units[key])
                return float(imbalance), utilization, key
            selected = min(feasible, key=score)
            output[selected].append(length)
            allocations[selected] += needed_units
            remaining[selected] -= needed_units
            for axis, bucket in zip(AXES, selected.split("|")[1:], strict=True):
                if bucket != "zero":
                    marginal[axis][bucket] += length
        diagnostics[mode] = {
            "target_steps": target,
            "selected_strata": dict(sorted((key, sum(output[key])) for key in candidates if output[key])),
            "marginal_axis_steps": {axis: dict(sorted(rows.items())) for axis, rows in marginal.items()},
        }
    return dict(output), {"common_capacity_units": common_units, "modes": diagnostics}


def main() -> None:
    args = parse_args()
    datasets = {}
    summaries = {}
    paths = {}
    for item in args.dataset:
        name, sep, raw = item.partition("=")
        if not sep:
            raise ValueError("--dataset must be NAME=PATH")
        path = Path(raw).expanduser()
        datasets[name], summaries[name] = capacities(path, args)
        paths[name] = str(path.resolve())
    blocks, diagnostics = choose_blocks(datasets, int(args.minimum_block))
    payload = {
        "schema": "go2_common_command_blocks_v2",
        "dataset_paths": paths,
        "mode_targets": MODE_TARGETS,
        "total_steps": sum(MODE_TARGETS.values()),
        "minimum_block": int(args.minimum_block),
        "zero_epsilon": float(args.zero_epsilon),
        "bin_edges": {"x": args.x_edges, "y": args.y_edges, "yaw": args.yaw_edges},
        "expert_only": bool(args.expert_only),
        "expert_collector_id": int(args.expert_collector_id),
        "blocks_by_stratum": {key: value for key, value in sorted(blocks.items())},
        "capacity_summaries": summaries,
        "diagnostics": diagnostics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.resolve()), "strata": len(blocks), "steps": sum(sum(v) for v in blocks.values())}, indent=2))


if __name__ == "__main__":
    main()
