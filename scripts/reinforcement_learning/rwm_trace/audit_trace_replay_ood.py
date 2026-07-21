#!/usr/bin/env python3
"""Bounded-memory OOD audit for TRACE replay transitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch


COMMAND_MODES = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")
STATE_NAMES = (
    *(f"base_lin_vel_{axis}" for axis in "xyz"),
    *(f"base_ang_vel_{axis}" for axis in "xyz"),
    *(f"projected_gravity_{axis}" for axis in "xyz"),
    *(f"joint_pos_{index}" for index in range(12)),
    *(f"joint_vel_{index}" for index in range(12)),
)
OBSERVATION_NAMES = (
    *STATE_NAMES,
    "command_vx",
    "command_vy",
    "command_yaw",
    *(f"previous_action_{index}" for index in range(12)),
)
ACTION_NAMES = tuple(f"action_{index}" for index in range(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_rows", type=int, default=16384)
    parser.add_argument("--reservoir_rows", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_mean_abs_z", type=float, default=5.0)
    parser.add_argument("--max_outside_fraction", type=float, default=0.5)
    parser.add_argument("--max_feature_mean_abs_z", type=float, default=10.0)
    parser.add_argument("--max_feature_outside_fraction", type=float, default=0.9)
    parser.add_argument("--max_command_mode_fraction_delta", type=float, default=0.20)
    parser.add_argument("--command_mode_presence_threshold", type=float, default=0.02)
    return parser.parse_args()


def _load_mmap(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the required fields exists: {keys}")


def _parts(value: Any) -> list[torch.Tensor]:
    if isinstance(value, (list, tuple)):
        return [torch.as_tensor(item) for item in value]
    return [torch.as_tensor(value)]


def _flat_chunks(value: Any, width: int, chunk_rows: int) -> Iterator[torch.Tensor]:
    for part in _parts(value):
        flat = part.reshape(-1, width)
        for start in range(0, len(flat), chunk_rows):
            yield flat[start : start + chunk_rows].float()


def _stored_field_with_width(data: dict[str, Any], width: int, *keys: str) -> Any | None:
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        parts = _parts(value)
        if parts and all(part.ndim >= 1 and int(part.shape[-1]) == width for part in parts):
            return value
    return None


def _dataset_observation_chunks(data: dict[str, Any], chunk_rows: int) -> Iterator[torch.Tensor]:
    stored = _stored_field_with_width(data, 48, "observation", "observations")
    if stored is not None:
        yield from _flat_chunks(stored, 48, chunk_rows)
        return
    states = _parts(data["states"])
    commands = _parts(data["commands"])
    previous = _parts(data["prev_actions"])
    if not (len(states) == len(commands) == len(previous)):
        raise ValueError("Dataset states/commands/prev_actions part counts differ.")
    for state_part, command_part, previous_part in zip(states, commands, previous, strict=True):
        state = state_part.reshape(-1, 45)
        command = command_part.reshape(-1, 3)
        prev = previous_part.reshape(-1, 12)
        if not (len(state) == len(command) == len(prev)):
            raise ValueError("Dataset observation components have different row counts.")
        for start in range(0, len(state), chunk_rows):
            stop = start + chunk_rows
            yield torch.cat((state[start:stop, :33], command[start:stop], prev[start:stop]), dim=-1).float()


def _dataset_action_chunks(data: dict[str, Any], chunk_rows: int) -> Iterator[torch.Tensor]:
    yield from _flat_chunks(_first(data, "action", "actions"), 12, chunk_rows)


def _dataset_next_observation_chunks(data: dict[str, Any], chunk_rows: int) -> Iterator[torch.Tensor]:
    stored = _stored_field_with_width(data, 48, "next_observation", "next_observations")
    if stored is not None:
        yield from _flat_chunks(stored, 48, chunk_rows)
        return
    states = _parts(data["next_states"])
    commands = _parts(data["commands"])
    actions = _parts(_first(data, "action", "actions"))
    if not (len(states) == len(commands) == len(actions)):
        raise ValueError("Dataset next_states/commands/actions part counts differ.")
    for state_part, command_part, action_part in zip(states, commands, actions, strict=True):
        state = state_part.reshape(-1, 45)
        command = command_part.reshape(-1, 3)
        action = action_part.reshape(-1, 12)
        if not (len(state) == len(command) == len(action)):
            raise ValueError("Dataset next-observation components have different row counts.")
        for start in range(0, len(state), chunk_rows):
            stop = start + chunk_rows
            # The current action becomes last_action in the next policy observation.
            yield torch.cat((state[start:stop, :33], command[start:stop], action[start:stop]), dim=-1).float()


class Reservoir:
    def __init__(self, capacity: int, width: int, seed: int):
        self.capacity = int(capacity)
        self.width = int(width)
        self.rng = np.random.default_rng(seed)
        self.rows = np.empty((self.capacity, self.width), dtype=np.float32)
        self.seen = 0
        self.size = 0

    def add(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        for row in values:
            self.seen += 1
            if self.size < self.capacity:
                self.rows[self.size] = row
                self.size += 1
            else:
                index = int(self.rng.integers(self.seen))
                if index < self.capacity:
                    self.rows[index] = row

    def values(self) -> np.ndarray:
        return self.rows[: self.size]


def _reference_stats(
    chunks: Callable[[], Iterator[torch.Tensor]],
    width: int,
    reservoir_rows: int,
    seed: int,
) -> dict[str, Any]:
    count = 0
    total = torch.zeros(width, dtype=torch.float64)
    total_sq = torch.zeros(width, dtype=torch.float64)
    reservoir = Reservoir(reservoir_rows, width, seed)
    for chunk in chunks():
        if chunk.shape[-1] != width or not bool(torch.isfinite(chunk).all()):
            raise ValueError(f"Reference dataset contains invalid {width}D values.")
        count += len(chunk)
        total += chunk.double().sum(0)
        total_sq += chunk.double().square().sum(0)
        reservoir.add(chunk)
    if count < 2:
        raise ValueError("Reference dataset contains fewer than two rows.")
    mean = total / count
    variance = ((total_sq - count * mean.square()) / (count - 1)).clamp_min(1.0e-12)
    sample = reservoir.values()
    return {
        "count": count,
        "mean": mean,
        "std": variance.sqrt(),
        "low": torch.from_numpy(np.quantile(sample, 0.01, axis=0)).double(),
        "high": torch.from_numpy(np.quantile(sample, 0.99, axis=0)).double(),
        "reservoir_rows": int(reservoir.size),
    }


def _audit_field(
    name: str,
    feature_names: tuple[str, ...],
    reference: dict[str, Any],
    replay_chunks: Iterator[torch.Tensor],
    reservoir_rows: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    width = len(feature_names)
    count = 0
    z_sum = torch.zeros(width, dtype=torch.float64)
    outside_count = torch.zeros(width, dtype=torch.int64)
    reservoir = Reservoir(reservoir_rows, width, seed)
    for chunk in replay_chunks:
        if chunk.shape[-1] != width or not bool(torch.isfinite(chunk).all()):
            raise ValueError(f"TRACE replay contains invalid {name} values.")
        values = chunk.double()
        z = ((values - reference["mean"]) / reference["std"]).abs()
        outside = (values < reference["low"]) | (values > reference["high"])
        count += len(chunk)
        z_sum += z.sum(0)
        outside_count += outside.sum(0)
        reservoir.add(z.float())
    if count == 0:
        raise ValueError(f"TRACE replay contains no {name} rows.")
    mean_z = (z_sum / count).cpu().numpy()
    outside = (outside_count.double() / count).cpu().numpy()
    sample = reservoir.values()
    p95 = np.quantile(sample, 0.95, axis=0)
    per_feature = [
        {
            "index": index,
            "name": feature_names[index],
            "mean_abs_z": float(mean_z[index]),
            "p95_abs_z": float(p95[index]),
            "outside_reference_1_99": float(outside[index]),
        }
        for index in range(width)
    ]
    worst_mean = max(per_feature, key=lambda row: row["mean_abs_z"])
    worst_outside = max(per_feature, key=lambda row: row["outside_reference_1_99"])
    aggregate_mean = float(mean_z.mean())
    aggregate_outside = float(outside.mean())
    passed = (
        math.isfinite(aggregate_mean)
        and aggregate_mean <= args.max_mean_abs_z
        and aggregate_outside <= args.max_outside_fraction
        and worst_mean["mean_abs_z"] <= args.max_feature_mean_abs_z
        and worst_outside["outside_reference_1_99"] <= args.max_feature_outside_fraction
    )
    return {
        "reference_rows": int(reference["count"]),
        "replay_rows": count,
        "reference_quantile_sample_rows": int(reference["reservoir_rows"]),
        "replay_quantile_sample_rows": int(reservoir.size),
        "mean_abs_z": aggregate_mean,
        "outside_reference_1_99": aggregate_outside,
        "worst_feature_by_mean_abs_z": worst_mean,
        "worst_feature_by_outside_fraction": worst_outside,
        "per_feature": per_feature,
        "passed": passed,
    }


def _command_mode_counts(chunks: Iterator[torch.Tensor], epsilon: float = 1.0e-3) -> dict[str, int]:
    counts = {mode: 0 for mode in COMMAND_MODES}
    for observations in chunks:
        commands = observations[:, 33:36].detach().cpu().numpy()
        x = np.abs(commands[:, 0]) > epsilon
        y = np.abs(commands[:, 1]) > epsilon
        yaw = np.abs(commands[:, 2]) > epsilon
        names = np.full(len(commands), "stand", dtype=object)
        names[x & ~y & ~yaw] = "pure_x"
        names[~x & y & ~yaw] = "pure_y"
        names[~x & ~y & yaw] = "pure_yaw"
        names[x & y & ~yaw] = "xy"
        names[x & ~y & yaw] = "x_yaw"
        names[~x & y & yaw] = "y_yaw"
        names[x & y & yaw] = "xy_yaw"
        unique, frequency = np.unique(names, return_counts=True)
        for mode, value in zip(unique, frequency, strict=True):
            counts[str(mode)] += int(value)
    return counts


def _command_mode_report(
    reference_counts: dict[str, int],
    replay_counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference_total = max(sum(reference_counts.values()), 1)
    replay_total = max(sum(replay_counts.values()), 1)
    rows = {}
    missing = []
    max_delta = 0.0
    for mode in COMMAND_MODES:
        reference_fraction = reference_counts[mode] / reference_total
        replay_fraction = replay_counts[mode] / replay_total
        delta = abs(replay_fraction - reference_fraction)
        max_delta = max(max_delta, delta)
        if reference_fraction >= args.command_mode_presence_threshold and replay_counts[mode] == 0:
            missing.append(mode)
        rows[mode] = {
            "reference_count": reference_counts[mode],
            "replay_count": replay_counts[mode],
            "reference_fraction": reference_fraction,
            "replay_fraction": replay_fraction,
            "absolute_fraction_delta": delta,
        }
    return {
        "per_mode": rows,
        "maximum_absolute_fraction_delta": max_delta,
        "missing_reference_modes": missing,
        "passed": not missing and max_delta <= args.max_command_mode_fraction_delta,
    }


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1 or args.reservoir_rows < 100:
        raise ValueError("chunk_rows must be positive and reservoir_rows must be at least 100.")
    dataset = _load_mmap(args.dataset)
    replay = _load_mmap(args.replay)
    field_specs = (
        (
            "observation",
            OBSERVATION_NAMES,
            lambda: _dataset_observation_chunks(dataset, args.chunk_rows),
            _flat_chunks(_first(replay, "observation", "observations"), 48, args.chunk_rows),
        ),
        (
            "action",
            ACTION_NAMES,
            lambda: _dataset_action_chunks(dataset, args.chunk_rows),
            _flat_chunks(_first(replay, "action", "actions"), 12, args.chunk_rows),
        ),
        (
            "next_observation",
            OBSERVATION_NAMES,
            lambda: _dataset_next_observation_chunks(dataset, args.chunk_rows),
            _flat_chunks(_first(replay, "next_observation", "next_observations"), 48, args.chunk_rows),
        ),
    )
    fields = {}
    for offset, (name, feature_names, reference_chunks, replay_chunks) in enumerate(field_specs):
        reference = _reference_stats(
            reference_chunks,
            len(feature_names),
            args.reservoir_rows,
            args.seed + 10 * offset,
        )
        fields[name] = _audit_field(
            name,
            feature_names,
            reference,
            replay_chunks,
            args.reservoir_rows,
            args.seed + 10 * offset + 1,
            args,
        )
    reference_modes = _command_mode_counts(_dataset_observation_chunks(dataset, args.chunk_rows))
    replay_modes = _command_mode_counts(
        _flat_chunks(_first(replay, "observation", "observations"), 48, args.chunk_rows)
    )
    command_modes = _command_mode_report(reference_modes, replay_modes, args)
    row_counts = {name: int(report["replay_rows"]) for name, report in fields.items()}
    row_count_consistent = len(set(row_counts.values())) == 1
    checks = {
        "observation": bool(fields["observation"]["passed"]),
        "action": bool(fields["action"]["passed"]),
        "next_observation": bool(fields["next_observation"]["passed"]),
        "command_modes": bool(command_modes["passed"]),
        "replay_row_counts_consistent": row_count_consistent,
    }
    report = {
        "schema": "go2_trace_replay_ood_v3",
        "dataset": str(Path(args.dataset).resolve()),
        "replay": str(Path(args.replay).resolve()),
        "thresholds": {
            "max_mean_abs_z": args.max_mean_abs_z,
            "max_outside_fraction": args.max_outside_fraction,
            "max_feature_mean_abs_z": args.max_feature_mean_abs_z,
            "max_feature_outside_fraction": args.max_feature_outside_fraction,
            "max_command_mode_fraction_delta": args.max_command_mode_fraction_delta,
            "command_mode_presence_threshold": args.command_mode_presence_threshold,
        },
        "fields": fields,
        "command_modes": command_modes,
        "replay_row_counts": row_counts,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".new")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
