"""Export exact signed magnitude-bin counts from a selected Go2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from select_go2_matched_blocks_v2 import time_env
from select_go2_matched_intervals_v2 import labels_and_marginals


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zero-epsilon", type=float, default=1e-3)
    parser.add_argument("--x-edges", type=float, nargs=4, default=(0.05, 0.20, 0.35, 0.50))
    parser.add_argument("--y-edges", type=float, nargs=4, default=(0.03, 0.087, 0.143, 0.20))
    parser.add_argument("--yaw-edges", type=float, nargs=4, default=(0.05, 0.167, 0.283, 0.40))
    args = parser.parse_args()
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    modes, marginals = labels_and_marginals(time_env(data["commands"], 3), args)
    mode_counts = np.bincount(modes.reshape(-1), minlength=8)
    payload = {
        "schema": "go2_condition_marginal_target_v2",
        "source_dataset": str(Path(args.dataset).resolve()),
        "mode_counts": mode_counts.tolist(),
        "marginal_target_counts": marginals.reshape(-1, 18).sum(0).tolist(),
        "zero_epsilon": float(args.zero_epsilon),
        "bin_edges": {"x": args.x_edges, "y": args.y_edges, "yaw": args.yaw_edges},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
