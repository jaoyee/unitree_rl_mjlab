"""Audit command/support coverage without changing a Go2 RWM dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


MODE = {
    (False, False, False): "stand",
    (True, False, False): "pure_x",
    (False, True, False): "pure_y",
    (False, False, True): "pure_yaw",
    (True, True, False): "xy",
    (True, False, True): "x_yaw",
    (False, True, True): "y_yaw",
    (True, True, True): "xy_yaw",
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
    parser.add_argument("--sequence-length", type=int, default=40)
    return parser.parse_args()


def stack(value: Any, key: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} is not a non-empty tensor/list")
    return torch.stack(value)


def as_time_env(value: Any, key: str, feature_dim: int | None = None) -> torch.Tensor:
    tensor = stack(value, key)
    if feature_dim is None:
        while tensor.ndim > 2 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.ndim != 2:
            raise ValueError(f"{key} must resolve to [T,E], got {tuple(tensor.shape)}")
        return tensor
    if tensor.shape[-1] != feature_dim:
        raise ValueError(f"{key} final dimension is {tensor.shape[-1]}, expected {feature_dim}")
    if tensor.ndim == 2:
        tensor = tensor[:, None, :]
    if tensor.ndim != 3:
        raise ValueError(f"{key} must resolve to [T,E,D], got {tuple(tensor.shape)}")
    return tensor


def axis_bin(value: float, edges: tuple[float, ...], epsilon: float) -> str | None:
    magnitude = abs(value)
    if magnitude <= epsilon:
        return "zero"
    if magnitude < edges[0] or magnitude > edges[-1] + 1e-6:
        return None
    index = min(2, sum(magnitude >= edge for edge in edges[1:-1]))
    return ("pos" if value > 0 else "neg") + f"_b{index}"


def classify(command: torch.Tensor, edges: tuple[tuple[float, ...], ...], epsilon: float) -> tuple[str, str] | None:
    bins = tuple(axis_bin(float(command[i]), edges[i], epsilon) for i in range(3))
    if any(item is None for item in bins):
        return None
    active = tuple(item != "zero" for item in bins)
    return MODE[active], "|".join((MODE[active], *bins))


def audit(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    commands = as_time_env(data["commands"], "commands", 3)
    episode = as_time_env(data["episode_ids"], "episode_ids")
    timestep = as_time_env(data["timesteps"], "timesteps")
    if episode.shape != commands.shape[:2] or timestep.shape != commands.shape[:2]:
        raise ValueError("command/episode/timestep shapes do not agree")
    eligible = torch.ones(commands.shape[:2], dtype=torch.bool)
    collector = None
    if "collector_types" in data:
        collector = as_time_env(data["collector_types"], "collector_types")
    if args.expert_only:
        if collector is None:
            raise ValueError("--expert-only requires collector_types")
        eligible &= collector == int(args.expert_collector_id)

    edges = (tuple(args.x_edges), tuple(args.y_edges), tuple(args.yaw_edges))
    labels: list[list[tuple[str, str] | None]] = []
    mode_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    marginal_counts = {axis: Counter() for axis in AXES}
    invalid_range = 0
    for t in range(commands.shape[0]):
        row: list[tuple[str, str] | None] = []
        for env in range(commands.shape[1]):
            label = classify(commands[t, env], edges, float(args.zero_epsilon))
            if label is None:
                invalid_range += int(eligible[t, env])
                eligible[t, env] = False
            elif eligible[t, env]:
                mode, stratum = label
                mode_counts[mode] += 1
                stratum_counts[stratum] += 1
                parts = stratum.split("|")[1:]
                for axis, part in zip(AXES, parts, strict=True):
                    marginal_counts[axis][part] += 1
            row.append(label)
        labels.append(row)

    run_lengths: list[int] = []
    mode_run_lengths: dict[str, list[int]] = {name: [] for name in MODE.values()}
    for env in range(commands.shape[1]):
        run_start = 0
        current_mode: str | None = None
        for t in range(commands.shape[0] + 1):
            mode = labels[t][env][0] if t < commands.shape[0] and eligible[t, env] else None
            contiguous = (
                t > run_start
                and mode == current_mode
                and int(episode[t, env]) == int(episode[t - 1, env])
                and int(timestep[t, env]) == int(timestep[t - 1, env]) + 1
            ) if t < commands.shape[0] else False
            if t == run_start:
                current_mode = mode
                continue
            if contiguous:
                continue
            if current_mode is not None:
                length = t - run_start
                run_lengths.append(length)
                mode_run_lengths[current_mode].append(length)
            run_start = t
            current_mode = mode

    horizon = int(args.sequence_length)
    return {
        "path": str(path.resolve()),
        "format_version": data.get("format_version"),
        "shape_time_env": list(commands.shape[:2]),
        "candidate_count": int(eligible.sum()),
        "invalid_range_count": invalid_range,
        "mode_counts": dict(sorted(mode_counts.items())),
        "joint_stratum_counts": dict(sorted(stratum_counts.items())),
        "marginal_axis_counts": {k: dict(sorted(v.items())) for k, v in marginal_counts.items()},
        "run_count": len(run_lengths),
        "run_length_min": min(run_lengths, default=0),
        "run_length_max": max(run_lengths, default=0),
        "valid_sequence_starts": sum(max(0, length - horizon + 1) for length in run_lengths),
        "mode_valid_sequence_starts": {
            mode: sum(max(0, length - horizon + 1) for length in lengths)
            for mode, lengths in mode_run_lengths.items()
        },
        "metadata": data.get("metadata", {}),
    }


def main() -> None:
    args = parse_args()
    reports = {}
    for item in args.dataset:
        name, separator, raw_path = item.partition("=")
        if not separator:
            raise ValueError("--dataset must be NAME=PATH")
        reports[name] = audit(Path(raw_path).expanduser(), args)
    output = {
        "schema": "go2_command_support_audit_v2",
        "expert_only": bool(args.expert_only),
        "expert_collector_id": int(args.expert_collector_id),
        "zero_epsilon": float(args.zero_epsilon),
        "bin_edges": {"x": args.x_edges, "y": args.y_edges, "yaw": args.yaw_edges},
        "sequence_length": int(args.sequence_length),
        "datasets": reports,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({name: {
        "candidate_count": report["candidate_count"],
        "mode_counts": report["mode_counts"],
        "valid_sequence_starts": report["valid_sequence_starts"],
    } for name, report in reports.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
