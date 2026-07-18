"""Select an exact, command-balanced subset from a converted real Go2 dataset.

The selector keeps each transition intact, never joins non-adjacent source
transitions into one episode, and prefers samples in the interior of long
strictly-valid contiguous spans so the offline sequence sampler retains useful
40-step windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


MODE_NAMES = (
    "stand",
    "pure_x",
    "pure_y",
    "pure_yaw",
    "xy",
    "x_yaw",
    "y_yaw",
    "xy_yaw",
)
MODE_BITS_TO_NAME = {
    (False, False, False): "stand",
    (True, False, False): "pure_x",
    (False, True, False): "pure_y",
    (False, False, True): "pure_yaw",
    (True, True, False): "xy",
    (True, False, True): "x_yaw",
    (False, True, True): "y_yaw",
    (True, True, True): "xy_yaw",
}
DEFAULT_TARGETS = {
    "stand": 2_000,
    "pure_x": 6_250,
    "pure_y": 2_500,
    "pure_yaw": 2_000,
    "xy": 3_500,
    "x_yaw": 4_250,
    "y_yaw": 1_250,
    "xy_yaw": 3_250,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--zero-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.05, 0.5))
    parser.add_argument("--y-range", type=float, nargs=2, default=(0.03, 0.2))
    parser.add_argument("--yaw-range", type=float, nargs=2, default=(0.05, 0.4))
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--condition-id", default="g0_baseline")
    return parser.parse_args()


def _stack(values: Any, key: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    if not isinstance(values, list) or not values:
        raise ValueError(f"dataset key {key!r} must be a non-empty list or tensor")
    return torch.stack(values, dim=0)


def _scalar_series(dataset: dict[str, Any], key: str, length: int) -> torch.Tensor:
    values = _stack(dataset[key], key).reshape(length, -1)
    if values.shape[1] != 1:
        raise ValueError(f"dataset key {key!r} must contain one scalar per transition")
    return values[:, 0]


def _classify_commands(
    commands: torch.Tensor,
    zero_epsilon: float,
    ranges: tuple[tuple[float, float], ...],
) -> tuple[list[str | None], torch.Tensor]:
    labels: list[str | None] = []
    eligible = torch.zeros(commands.shape[0], dtype=torch.bool)
    for index, command in enumerate(commands.tolist()):
        active = tuple(abs(float(value)) > zero_epsilon for value in command)
        valid = True
        for axis, value in enumerate(command):
            magnitude = abs(float(value))
            if active[axis]:
                lower, upper = ranges[axis]
                valid = valid and lower <= magnitude <= upper + 1.0e-6
        if valid:
            labels.append(MODE_BITS_TO_NAME[active])
            eligible[index] = True
        else:
            labels.append(None)
    return labels, eligible


def _valid_segments(
    eligible: torch.Tensor,
    episode_ids: torch.Tensor,
    timesteps: torch.Tensor,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(int(eligible.numel())):
        contiguous = (
            start is not None
            and bool(eligible[index])
            and int(episode_ids[index]) == int(episode_ids[index - 1])
            and int(timesteps[index]) == int(timesteps[index - 1]) + 1
        )
        if bool(eligible[index]) and (start is None or contiguous):
            if start is None:
                start = index
            continue
        if start is not None:
            segments.append((start, index))
            start = None
        if bool(eligible[index]):
            start = index
    if start is not None:
        segments.append((start, int(eligible.numel())))
    return segments


def _select_exact(
    labels: list[str | None],
    segments: list[tuple[int, int]],
    targets: dict[str, int],
) -> torch.Tensor:
    candidates: dict[str, list[tuple[int, int, int]]] = {name: [] for name in MODE_NAMES}
    for start, end in segments:
        length = end - start
        for index in range(start, end):
            label = labels[index]
            if label is None:
                continue
            edge_distance = min(index - start + 1, end - index)
            candidates[label].append((length, edge_distance, index))

    selected = torch.zeros(len(labels), dtype=torch.bool)
    for name in MODE_NAMES:
        need = int(targets[name])
        ranked = sorted(candidates[name], key=lambda item: (item[0], item[1]), reverse=True)
        if len(ranked) < need:
            raise ValueError(f"mode {name!r} has {len(ranked)} eligible transitions; need {need}")
        for _, _, index in ranked[:need]:
            selected[index] = True
    if int(selected.sum()) != sum(targets.values()):
        raise RuntimeError("selected transition count does not equal requested total")
    return selected


def _selected_runs(
    selected_indices: list[int],
    episode_ids: torch.Tensor,
    timesteps: torch.Tensor,
) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in selected_indices:
        if not runs:
            runs.append([index])
            continue
        previous = runs[-1][-1]
        contiguous = (
            index == previous + 1
            and int(episode_ids[index]) == int(episode_ids[previous])
            and int(timesteps[index]) == int(timesteps[previous]) + 1
        )
        if contiguous:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _filter_dataset(
    dataset: dict[str, Any],
    selected_indices: list[int],
    runs: list[list[int]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    transition_count = len(dataset["states"])
    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    for key, value in list(dataset.items()):
        if isinstance(value, list) and len(value) == transition_count:
            output[key] = [value[index] for index in selected_indices]
        elif isinstance(value, torch.Tensor) and value.shape[0] == transition_count:
            output[key] = value.index_select(0, index_tensor)
        elif key == "metadata":
            output[key] = dict(value or {})
        else:
            output[key] = value

    new_episode_ids: list[torch.Tensor] = []
    new_timesteps: list[torch.Tensor] = []
    for episode_id, run in enumerate(runs):
        for timestep, _ in enumerate(run):
            new_episode_ids.append(torch.tensor([episode_id], dtype=torch.long))
            new_timesteps.append(torch.tensor([timestep], dtype=torch.long))
    output["episode_ids"] = new_episode_ids
    output["timesteps"] = new_timesteps
    output["capacity"] = len(selected_indices)
    return output


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser()
    input_sha256 = _sha256(input_path)
    source = torch.load(input_path, map_location="cpu", weights_only=False)
    if int(source.get("num_envs", 0)) != 1:
        raise ValueError("real Go2 selection currently requires num_envs=1")
    transition_count = len(source["states"])
    commands = _stack(source["commands"], "commands").reshape(transition_count, -1)
    if commands.shape[1] != 3:
        raise ValueError(f"expected 3 command dimensions, got {commands.shape[1]}")
    episode_ids = _scalar_series(source, "episode_ids", transition_count).long()
    timesteps = _scalar_series(source, "timesteps", transition_count).long()
    ranges = (tuple(args.x_range), tuple(args.y_range), tuple(args.yaw_range))
    labels, eligible = _classify_commands(commands, float(args.zero_epsilon), ranges)
    available = {name: labels.count(name) for name in MODE_NAMES}
    segments = _valid_segments(eligible, episode_ids, timesteps)
    selected = _select_exact(labels, segments, DEFAULT_TARGETS)
    selected_indices = selected.nonzero(as_tuple=False).flatten().tolist()
    runs = _selected_runs(selected_indices, episode_ids, timesteps)
    output = _filter_dataset(source, selected_indices, runs)

    selected_counts = {
        name: sum(labels[index] == name for index in selected_indices) for name in MODE_NAMES
    }
    if selected_counts != DEFAULT_TARGETS:
        raise RuntimeError(f"selected command counts do not match targets: {selected_counts}")
    run_lengths = [len(run) for run in runs]
    sequence_count = sum(max(0, length - int(args.sequence_length) + 1) for length in run_lengths)
    metadata = dict(output.get("metadata") or {})
    metadata.update(
        {
            "condition_id": args.condition_id,
            "selection_kind": "strict_exact_command_coverage",
            "selection_source": str(input_path.resolve()),
            "selection_source_sha256": input_sha256,
            "num_time_steps": len(selected_indices),
            "num_transitions": len(selected_indices),
            "num_episodes": len(runs),
            "command_mode_counts": selected_counts,
            "command_mode_weights": {
                name: selected_counts[name] / len(selected_indices) for name in MODE_NAMES
            },
            "command_ranges": {
                "lin_vel_x_abs": list(args.x_range),
                "lin_vel_y_abs": list(args.y_range),
                "ang_vel_z_abs": list(args.yaw_range),
                "zero_epsilon": float(args.zero_epsilon),
            },
            "selection_available_counts": available,
            "selection_invalid_transitions": int((~eligible).sum()),
            "selection_contiguous_runs": len(runs),
            "selection_sequence_length": int(args.sequence_length),
            "selection_valid_sequence_starts": sequence_count,
        }
    )
    output["metadata"] = metadata

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report = {
        "input": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "output": str(output_path.resolve()),
        "condition_id": args.condition_id,
        "input_transitions": transition_count,
        "eligible_transitions": int(eligible.sum()),
        "invalid_transitions": int((~eligible).sum()),
        "available_counts": available,
        "selected_transitions": len(selected_indices),
        "selected_counts": selected_counts,
        "selected_weights": metadata["command_mode_weights"],
        "selected_runs": len(runs),
        "run_length_min": min(run_lengths),
        "run_length_max": max(run_lengths),
        "run_length_mean": sum(run_lengths) / len(run_lengths),
        "sequence_length": int(args.sequence_length),
        "valid_sequence_starts": sequence_count,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
