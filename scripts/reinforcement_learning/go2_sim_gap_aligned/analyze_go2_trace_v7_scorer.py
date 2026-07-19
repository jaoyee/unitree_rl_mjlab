#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import stack_time_key
from scripts.reinforcement_learning.rwm_flashsac.dynamics_loader import (
    load_any_go2_dynamics_checkpoint,
)
from scripts.reinforcement_learning.rwm_trace.build_trace_replay_v4 import _candidate_windows


def finite_spearman(rows, key):
    x = np.asarray([r.get("trace_score", np.nan) for r in rows], dtype=float)
    y = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    return float(spearmanr(x[valid], y[valid]).statistic)


def mean(rows, key):
    values = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def command_mode(command, eps=1e-6):
    active = tuple(abs(float(v)) > eps for v in command)
    return {
        (False, False, False): "stand",
        (True, False, False): "pure_x",
        (False, True, False): "pure_y",
        (False, False, True): "pure_yaw",
        (True, True, False): "xy",
        (True, False, True): "x_yaw",
        (False, True, True): "y_yaw",
        (True, True, True): "xy_yaw",
    }[active]


def command_distribution(candidate_path, rows):
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    commands = stack_time_key(payload, "commands")
    counts = {"candidate": {}, "top25": {}}
    for row in rows:
        fields = row["trajectory_id"].split("_")
        env_id = int(fields[0].removeprefix("env"))
        start = int(fields[1].removeprefix("start"))
        mode = command_mode(commands[start, env_id].tolist())
        bucket = "top25" if row.get("selected", False) else None
        counts["candidate"][mode] = counts["candidate"].get(mode, 0) + 1
        if bucket:
            counts[bucket][mode] = counts[bucket].get(mode, 0) + 1
    for bucket in counts:
        total = sum(counts[bucket].values())
        counts[bucket] = {k: v / total for k, v in sorted(counts[bucket].items())} if total else {}
    return counts


@torch.no_grad()
def trajectory_uncertainty(candidate_path, dynamics, device, cache_path):
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    windows = _candidate_windows(
        payload,
        100,
        100,
        include_terminal_prefixes=True,
        minimum_terminal_length=20,
    )
    history = int(dynamics.cfg.history_horizon)
    result = {}
    for offset in range(0, len(windows), 64):
        batch = windows[offset : offset + 64]
        state_histories = []
        action_histories = []
        owners = []
        for owner, trajectory in enumerate(batch):
            states = trajectory["states"].float()
            actions = trajectory["actions"].float()
            for end in range(history, len(states) + 1):
                state_histories.append(states[end - history : end])
                action_histories.append(actions[end - history : end])
                owners.append(owner)
        sums = np.zeros(len(batch), dtype=np.float64)
        counts = np.zeros(len(batch), dtype=np.int64)
        for start in range(0, len(state_histories), 4096):
            stop = min(start + 4096, len(state_histories))
            states_t = torch.stack(state_histories[start:stop]).to(device)
            actions_t = torch.stack(action_histories[start:stop]).to(device)
            model_ids = torch.zeros(stop - start, dtype=torch.long, device=device)
            epistemic = dynamics.predict(states_t, actions_t, model_ids)[2].detach().cpu().numpy()
            for owner, value in zip(owners[start:stop], epistemic):
                sums[owner] += float(value)
                counts[owner] += 1
        for owner, trajectory in enumerate(batch):
            result[trajectory["trajectory_id"]] = (
                float(sums[owner] / counts[owner]) if counts[owner] else None
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=("g0", "rr03"))
    parser.add_argument("--branches", nargs="+", default=("r10", "r25"))
    parser.add_argument("--max-cycle", type=int, default=8)
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="Optional frozen RWM used to compute per-trajectory epistemic uncertainty.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = dict(item.split("=", 1) for item in args.model_path)

    results = []
    for condition in args.conditions:
        dynamics = None
        device = torch.device(args.device)
        if condition in model_paths:
            dynamics, _ = load_any_go2_dynamics_checkpoint(Path(model_paths[condition]), device=device)
            dynamics.eval()
        for branch in args.branches:
            branch_root = args.formal_root / "sim" / condition / f"trace_{branch}"
            for cycle in range(1, args.max_cycle + 1):
                refresh = branch_root / f"refresh_{cycle:02d}"
                summary_path = refresh / "replay" / f"{condition}.summaries.jsonl"
                if not summary_path.exists():
                    continue
                rows = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
                candidate_path = refresh / "candidates" / f"{condition}.pt"
                if dynamics is not None and candidate_path.exists():
                    uncertainty = trajectory_uncertainty(
                        candidate_path,
                        dynamics,
                        device,
                        args.output_dir / "uncertainty_cache" / condition / branch
                        / f"refresh_{cycle:02d}.json",
                    )
                    for row in rows:
                        row["epistemic_uncertainty"] = uncertainty.get(row["trajectory_id"])
                selected = [row for row in rows if row.get("selected", False)]
                selected_count = len(selected)
                oracle = sorted(
                    rows, key=lambda row: float(row.get("simulator_return", -np.inf)), reverse=True
                )[:selected_count]
                record = {
                    "condition": condition,
                    "branch": branch,
                    "refresh": cycle,
                    "candidate_count": len(rows),
                    "selected_count": len(selected),
                    "score_return_spearman": finite_spearman(rows, "simulator_return"),
                    "score_episode_length_spearman": finite_spearman(rows, "survival_length"),
                    "score_survival_spearman": finite_spearman(rows, "survival_length"),
                    "score_uncertainty_spearman": finite_spearman(rows, "epistemic_uncertainty"),
                    "score_xy_error_spearman": finite_spearman(rows, "linear_tracking_error_mean"),
                    "score_yaw_error_spearman": finite_spearman(rows, "yaw_tracking_error_mean"),
                    "candidate_terminal_ratio": mean(rows, "terminal_flag"),
                    "selected_terminal_ratio": mean(selected, "terminal_flag"),
                    "candidate_return_mean": mean(rows, "simulator_return"),
                    "selected_return_mean": mean(selected, "simulator_return"),
                    "candidate_xy_error_mean": mean(rows, "linear_tracking_error_mean"),
                    "selected_xy_error_mean": mean(selected, "linear_tracking_error_mean"),
                    "candidate_yaw_error_mean": mean(rows, "yaw_tracking_error_mean"),
                    "selected_yaw_error_mean": mean(selected, "yaw_tracking_error_mean"),
                    "candidate_uncertainty_mean": mean(rows, "epistemic_uncertainty"),
                    "selected_uncertainty_mean": mean(selected, "epistemic_uncertainty"),
                    "oracle_return_mean": mean(oracle, "simulator_return"),
                    "oracle_terminal_ratio": mean(oracle, "terminal_flag"),
                }
                if candidate_path.exists():
                    record["command_distribution"] = command_distribution(candidate_path, rows)
                results.append(record)

    (args.output_dir / "scorer_diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")
    if results:
        keys = sorted({key for row in results for key in row if key != "command_distribution"})
        with (args.output_dir / "scorer_diagnostics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in keys} for row in results)

    lines = [
        "# V7 scorer diagnostics",
        "",
        "Spearman signs: higher scorer scores should correlate positively with return/episode length and negatively with tracking errors.",
        "Uncertainty fields remain NA until model-based per-trajectory uncertainty is attached to the summaries.",
        "",
        "|condition|branch|refresh|rho(return)|rho(survival)|rho(uncert.)|rho(XY err)|rho(yaw err)|terminal all|terminal top25|return all|return top25|uncert. all|uncert. top25|oracle return|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        def fmt(value):
            return "NA" if value is None else f"{value:.4f}"
        lines.append(
            f"|{row['condition']}|{row['branch']}|{row['refresh']}|"
            f"{fmt(row['score_return_spearman'])}|{fmt(row['score_episode_length_spearman'])}|"
            f"{fmt(row['score_uncertainty_spearman'])}|"
            f"{fmt(row['score_xy_error_spearman'])}|{fmt(row['score_yaw_error_spearman'])}|"
            f"{fmt(row['candidate_terminal_ratio'])}|{fmt(row['selected_terminal_ratio'])}|"
            f"{fmt(row['candidate_return_mean'])}|{fmt(row['selected_return_mean'])}|"
            f"{fmt(row['candidate_uncertainty_mean'])}|{fmt(row['selected_uncertainty_mean'])}|"
            f"{fmt(row['oracle_return_mean'])}|"
        )
    (args.output_dir / "scorer_diagnostics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
