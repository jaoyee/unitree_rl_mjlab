#!/usr/bin/env python3
"""Append TRACE candidate branches along the vector-environment dimension."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base", required=True)
    parser.add_argument("--append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _concat_table(left: dict[str, Any], right: dict[str, Any], name: str) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError(f"{name} fields differ: {sorted(set(left) ^ set(right))}")
    result: dict[str, Any] = {}
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result[key] = torch.cat((a, b), dim=0)
        elif a == b:
            result[key] = a
        else:
            raise ValueError(f"Cannot merge non-tensor {name}.{key}")
    return result


def main() -> None:
    args = parse_args()
    base_path = Path(args.base).expanduser().resolve()
    append_path = Path(args.append).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    left = torch.load(base_path, map_location="cpu", weights_only=False)
    right = torch.load(append_path, map_location="cpu", weights_only=False)
    left_envs, right_envs = int(left["num_envs"]), int(right["num_envs"])

    left_lists = {key for key, value in left.items() if isinstance(value, list)}
    right_lists = {key for key, value in right.items() if isinstance(value, list)}
    if left_lists != right_lists:
        raise ValueError(f"Candidate time-series fields differ: {sorted(left_lists ^ right_lists)}")
    merged = dict(left)
    for key in sorted(left_lists):
        a_values, b_values = left[key], right[key]
        if len(a_values) != len(b_values):
            raise ValueError(f"Candidate time lengths differ for {key}: {len(a_values)} != {len(b_values)}")
        rows = []
        for step, (a, b) in enumerate(zip(a_values, b_values, strict=True)):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"Expected tensor rows for {key} at step {step}")
            if int(a.shape[0]) != left_envs or int(b.shape[0]) != right_envs:
                raise ValueError(f"Bad environment dimension for {key} at step {step}")
            if tuple(a.shape[1:]) != tuple(b.shape[1:]):
                raise ValueError(f"Feature shape differs for {key} at step {step}")
            rows.append(torch.cat((a, b), dim=0))
        merged[key] = rows

    for key in ("trace_start_state_ids",):
        a, b = torch.as_tensor(left[key]), torch.as_tensor(right[key])
        if a.numel() != left_envs or b.numel() != right_envs:
            raise ValueError(f"Bad {key} width while merging candidate branches")
        merged[key] = torch.cat((a.reshape(-1), b.reshape(-1)), dim=0)

    for key in ("startup_domain_table", "episode_domain_table"):
        if key in left or key in right:
            merged[key] = _concat_table(left.get(key, {}), right.get(key, {}), key)

    merged["num_envs"] = left_envs + right_envs
    metadata = dict(left.get("metadata") or {})
    metadata["branch_append_parts"] = int(metadata.get("branch_append_parts", 1)) + int(
        (right.get("metadata") or {}).get("branch_append_parts", 1)
    )
    metadata["branch_append_sources"] = [str(base_path), str(append_path)]
    valid = merged.get("trace_valid_masks")
    if isinstance(valid, list) and valid:
        actual = int(torch.stack(valid, dim=0).bool().sum())
    else:
        actual = len(merged["states"]) * int(merged["num_envs"])
    metadata["actual_num_transitions"] = actual
    metadata["rectangular_storage_rows"] = len(merged["states"]) * int(merged["num_envs"])
    metadata["num_transitions"] = actual
    merged["metadata"] = metadata
    merged["capacity"] = actual

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".new")
    torch.save(merged, temporary)
    temporary.replace(output_path)
    print(
        f"merged candidate branches: envs={left_envs}+{right_envs}={merged['num_envs']}, "
        f"valid_transitions={actual}, output={output_path}"
    )


if __name__ == "__main__":
    main()
