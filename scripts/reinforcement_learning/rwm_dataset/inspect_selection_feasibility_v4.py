#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_go2_matched_intervals_v2 import MODE_NAMES, labels_and_marginals
from select_go2_matched_blocks_v2 import time_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    commands = time_env(data["commands"], 3)

    class Settings:
        zero_epsilon = 1e-3
        x_edges = (0.001, 0.20, 0.35, 0.50)
        y_edges = (0.001, 0.087, 0.143, 0.20)
        yaw_edges = (0.001, 0.167, 0.283, 0.40)

    mode_ids, _ = labels_and_marginals(commands, Settings())
    flat_commands = commands.detach().cpu().numpy().reshape(-1, 3)
    absolute = np.abs(flat_commands)
    axis_limits = np.asarray((0.50, 0.20, 0.40))
    axis_minimum = np.asarray((0.001, 0.001, 0.001))
    active = absolute > Settings.zero_epsilon
    invalid_high = active & (absolute > axis_limits[None, :] + 1e-6)
    invalid_low = active & (absolute < axis_minimum[None, :])
    counts = {name: int((mode_ids == index).sum()) for index, name in enumerate(MODE_NAMES)}
    counts["invalid_or_out_of_range"] = int((mode_ids < 0).sum())
    runs = []
    for env in range(mode_ids.shape[1]):
        valid = mode_ids[:, env] >= 0
        start = None
        for t in range(len(valid) + 1):
            if t < len(valid) and valid[t] and start is None:
                start = t
            if start is not None and (t == len(valid) or not valid[t]):
                runs.append(t - start)
                start = None
    print(json.dumps({
        "valid_mode_counts": counts,
        "valid_runs": {
            "count": len(runs), "min": min(runs, default=0),
            "median": float(np.median(runs)) if runs else 0.0,
            "max": max(runs, default=0),
            "transitions_in_runs_ge_100": int(sum(length for length in runs if length >= 100)),
        },
        "command_abs_quantiles": {
            axis: [float(value) for value in np.quantile(absolute[:, index], [0.0, 0.5, 0.9, 0.99, 1.0])]
            for index, axis in enumerate(("x", "y", "yaw"))
        },
        "invalid_low_counts": dict(zip(("x", "y", "yaw"), invalid_low.sum(axis=0).astype(int).tolist())),
        "invalid_high_counts": dict(zip(("x", "y", "yaw"), invalid_high.sum(axis=0).astype(int).tolist())),
    }, indent=2))


if __name__ == "__main__":
    main()
