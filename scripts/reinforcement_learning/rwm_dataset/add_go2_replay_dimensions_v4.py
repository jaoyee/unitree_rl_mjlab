#!/usr/bin/env python3
"""Add SequenceReplayBuffer top-level dimension fields to a pooled dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    path = Path(args.input).expanduser().resolve()
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    dimensions = {
        "state_dim": int(torch.stack(dataset["states"]).shape[-1]),
        "action_dim": int(torch.stack(dataset["actions"]).shape[-1]),
        "contact_dim": int(torch.stack(dataset["contacts"]).shape[-1]),
        "termination_dim": int(torch.stack(dataset["terminations"]).shape[-1]),
    }
    metadata = dict(dataset.get("metadata") or {})
    for key, value in dimensions.items():
        metadata_value = metadata.get(key)
        if metadata_value is not None and int(metadata_value) != value:
            raise ValueError(f"{key} mismatch: tensor={value}, metadata={metadata_value}")
        dataset[key] = value
    dataset["metadata"] = metadata
    torch.save(dataset, path)
    print({"path": str(path), **dimensions, "capacity": dataset.get("capacity"), "num_envs": dataset.get("num_envs")})


if __name__ == "__main__":
    main()
