"""Extract episode-safe pre-fall windows from a converted real Go2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from select_go2_matched_blocks_v2 import filter_dataset, stack, time_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--history", type=int, default=40)
    parser.add_argument("--condition-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    terminations = time_env(data["terminations"])
    episodes = time_env(data["episode_ids"])
    timesteps = time_env(data["timesteps"])
    blocks = []
    for env in range(terminations.shape[1]):
        for end in (terminations[:, env] > 0.5).nonzero(as_tuple=False).flatten().tolist():
            start = end
            while start > 0 and end - start + 1 < args.history:
                if int(episodes[start - 1, env]) != int(episodes[end, env]):
                    break
                if int(timesteps[start, env]) != int(timesteps[start - 1, env]) + 1:
                    break
                start -= 1
            blocks.append([(t, env) for t in range(start, end + 1)])
    if not blocks:
        report = {
            "schema": "go2_termination_aux_v2", "condition_id": args.condition_id,
            "input": str(Path(args.input).resolve()), "terminal_events": 0,
            "transitions": 0, "status": "empty_no_recorded_termination",
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    output = filter_dataset(data, blocks, tuple(terminations.shape))
    metadata = dict(output.get("metadata") or {})
    metadata.update({
        "condition_id": args.condition_id,
        "selection_kind": "termination_pre_fall_aux_v2",
        "selection_source": str(Path(args.input).resolve()),
        "termination_history": int(args.history),
        "terminal_events": len(blocks),
        "num_transitions": sum(map(len, blocks)),
        "num_episodes": len(blocks),
    })
    output["metadata"] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    report = {
        "schema": "go2_termination_aux_v2", "condition_id": args.condition_id,
        "input": str(Path(args.input).resolve()), "output": str(output_path.resolve()),
        "terminal_events": len(blocks), "transitions": sum(map(len, blocks)),
        "block_lengths": list(map(len, blocks)), "status": "pass",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
