"""Select exact shared command strata as contiguous blocks from real or sim Go2 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


MODE = {
    (False, False, False): "stand", (True, False, False): "pure_x",
    (False, True, False): "pure_y", (False, False, True): "pure_yaw",
    (True, True, False): "xy", (True, False, True): "x_yaw",
    (False, True, True): "y_yaw", (True, True, True): "xy_yaw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--block-spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--expert-only", action="store_true")
    parser.add_argument("--expert-collector-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def candidate_segments(
    data: dict[str, Any], spec: dict[str, Any], expert_only: bool, expert_id: int
) -> tuple[dict[str, list[list[tuple[int, int]]]], tuple[int, int]]:
    commands = time_env(data["commands"], 3)
    episodes = time_env(data["episode_ids"])
    timesteps = time_env(data["timesteps"])
    terminations = time_env(data["terminations"])
    allowed = terminations <= 0.5
    if expert_only:
        allowed &= time_env(data["collector_types"]) == expert_id
    edges = tuple(tuple(spec["bin_edges"][axis]) for axis in ("x", "y", "yaw"))
    wanted = set(spec["blocks_by_stratum"])
    result: dict[str, list[list[tuple[int, int]]]] = defaultdict(list)
    for env in range(commands.shape[1]):
        start = 0
        current: str | None = None
        for t in range(commands.shape[0] + 1):
            label = stratum(commands[t, env], edges, float(spec["zero_epsilon"])) if t < commands.shape[0] and allowed[t, env] else None
            label = label if label in wanted else None
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
            if current is not None and t - start >= int(spec["minimum_block"]):
                result[current].append([(source_t, env) for source_t in range(start, t)])
            start, current = t, label
    return dict(result), tuple(commands.shape[:2])


def select_blocks(
    candidates: dict[str, list[list[tuple[int, int]]]],
    requested: dict[str, list[int]],
    seed: int,
) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[list[tuple[int, int]]] = []
    diagnostics = {}
    for key, lengths in sorted(requested.items()):
        pools = [list(segment) for segment in candidates.get(key, [])]
        rng.shuffle(pools)
        pools.sort(key=len, reverse=True)
        chosen_lengths = []
        for requested_length in sorted(lengths, reverse=True):
            pool_index = next((idx for idx, segment in enumerate(pools) if len(segment) >= requested_length), None)
            if pool_index is None:
                raise ValueError(f"stratum {key} cannot supply contiguous block length {requested_length}")
            segment = pools[pool_index]
            max_offset = len(segment) - requested_length
            offset = rng.randint(0, max_offset) if max_offset else 0
            block = segment[offset:offset + requested_length]
            selected.append(block)
            chosen_lengths.append(len(block))
            leftovers = [segment[:offset], segment[offset + requested_length:]]
            pools.pop(pool_index)
            pools.extend(part for part in leftovers if part)
            pools.sort(key=len, reverse=True)
        diagnostics[key] = {
            "requested_lengths": lengths,
            "selected_lengths": chosen_lengths,
            "candidate_segments": len(candidates.get(key, [])),
        }
    return selected, diagnostics


def transition_value(value: Any, t: int, env: int, time_steps: int, num_envs: int) -> torch.Tensor:
    tensor = value[t] if isinstance(value, list) else value[t]
    if num_envs == 1:
        if tensor.ndim > 0 and tensor.shape[0] == 1:
            return tensor[0].clone()
        return tensor.clone()
    return tensor[env].clone()


def filter_dataset(
    data: dict[str, Any], blocks: list[list[tuple[int, int]]], shape: tuple[int, int]
) -> dict[str, Any]:
    time_steps, num_envs = shape
    ordered = sorted(enumerate(blocks), key=lambda item: item[1][0])
    output: dict[str, Any] = {}
    transition_keys = []
    for key, value in data.items():
        if isinstance(value, list) and len(value) == time_steps:
            transition_keys.append(key)
        elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == time_steps:
            transition_keys.append(key)
        elif key == "metadata":
            output[key] = dict(value or {})
        else:
            output[key] = value
    for key in transition_keys:
        output[key] = []
    output["source_env_ids"] = []
    output["source_episode_ids"] = []
    output["source_timesteps"] = []
    output["source_time_indices"] = []
    source_episodes = time_env(data["episode_ids"])
    source_timesteps = time_env(data["timesteps"])
    new_episode_ids = []
    new_timesteps = []
    for new_episode, (_, block) in enumerate(ordered):
        for new_timestep, (t, env) in enumerate(block):
            for key in transition_keys:
                output[key].append(transition_value(data[key], t, env, time_steps, num_envs).unsqueeze(0))
            output["source_env_ids"].append(torch.tensor([env], dtype=torch.long))
            output["source_episode_ids"].append(torch.tensor([int(source_episodes[t, env])], dtype=torch.long))
            output["source_timesteps"].append(torch.tensor([int(source_timesteps[t, env])], dtype=torch.long))
            output["source_time_indices"].append(torch.tensor([t], dtype=torch.long))
            new_episode_ids.append(torch.tensor([new_episode], dtype=torch.long))
            new_timesteps.append(torch.tensor([new_timestep], dtype=torch.long))
    output["episode_ids"] = new_episode_ids
    output["timesteps"] = new_timesteps
    output["num_envs"] = 1
    output["capacity"] = len(new_episode_ids)
    return output


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    spec_path = Path(args.block_spec).expanduser()
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    spec = json.loads(spec_path.read_text())
    candidates, shape = candidate_segments(data, spec, args.expert_only, args.expert_collector_id)
    blocks, diagnostics = select_blocks(candidates, spec["blocks_by_stratum"], args.seed)
    output = filter_dataset(data, blocks, shape)
    total = sum(len(block) for block in blocks)
    if total != int(spec["total_steps"]):
        raise RuntimeError(f"selected {total}, expected {spec['total_steps']}")
    metadata = dict(output.get("metadata") or {})
    metadata.update({
        "condition_id": args.condition_id,
        "selection_kind": "shared_command_strata_contiguous_blocks_v2",
        "selection_source": str(input_path.resolve()),
        "selection_source_sha256": sha256(input_path),
        "selection_block_spec": str(spec_path.resolve()),
        "selection_block_spec_sha256": sha256(spec_path),
        "selection_seed": int(args.seed),
        "selection_expert_only": bool(args.expert_only),
        "selection_minimum_block": int(spec["minimum_block"]),
        "selection_block_count": len(blocks),
        "selection_block_lengths": [len(block) for block in blocks],
        "num_time_steps": total,
        "num_transitions": total,
        "num_episodes": len(blocks),
        "command_mode_counts": spec["mode_targets"],
        "command_block_spec": spec["blocks_by_stratum"],
    })
    output["metadata"] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report = {
        "schema": "go2_matched_block_selection_v2",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "condition_id": args.condition_id,
        "source_shape_time_env": list(shape),
        "selected_transitions": total,
        "selected_blocks": len(blocks),
        "block_length_min": min(map(len, blocks)),
        "block_length_max": max(map(len, blocks)),
        "valid_40_step_starts": sum(len(block) - 40 + 1 for block in blocks),
        "diagnostics": diagnostics,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("condition_id", "selected_transitions", "selected_blocks", "valid_40_step_starts")}, indent=2))


if __name__ == "__main__":
    main()
