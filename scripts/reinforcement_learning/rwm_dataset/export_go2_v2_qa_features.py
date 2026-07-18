"""Export compact tensors needed for paired sim-real Go2 dataset QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--domain", choices=("real", "sim"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    keys = ("states", "actions", "commands", "contacts", "terminations")
    tensors = {
        key: stack_time_key(dataset, key).reshape(len(dataset["states"]), -1).float()
        for key in keys
    }
    package = {
        **tensors,
        "condition_id": args.condition_id,
        "domain": args.domain,
        "source_dataset": str(dataset_path.resolve()),
        "source_metadata": dataset.get("metadata") or {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, output)
    print(json.dumps({
        "status": "pass",
        "condition_id": args.condition_id,
        "domain": args.domain,
        "output": str(output.resolve()),
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
