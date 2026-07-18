"""Compare one matched sim-real Go2 dataset pair without normalizing away the gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance


GROUPS = {
    "base_lin_vel": ("states", 0, 3),
    "base_ang_vel": ("states", 3, 6),
    "projected_gravity": ("states", 6, 9),
    "joint_pos_rel": ("states", 9, 21),
    "joint_vel": ("states", 21, 33),
    "actuator_force": ("states", 33, 45),
    "action": ("actions", 0, 12),
    "command": ("commands", 0, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--real", required=True)
    parser.add_argument("--sim", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def dimension_metrics(real: np.ndarray, sim: np.ndarray) -> dict[str, list[float]]:
    wasserstein = []
    ks = []
    for dim in range(real.shape[1]):
        wasserstein.append(float(wasserstein_distance(real[:, dim], sim[:, dim])))
        ks.append(float(ks_2samp(real[:, dim], sim[:, dim]).statistic))
    return {"wasserstein": wasserstein, "ks": ks}


def summary(values: np.ndarray) -> dict[str, object]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "quantiles": {
            str(q): np.quantile(values, q, axis=0).tolist()
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
    }


def main() -> None:
    args = parse_args()
    real = torch.load(args.real, map_location="cpu", weights_only=False)
    sim = torch.load(args.sim, map_location="cpu", weights_only=False)
    if real["condition_id"] != sim["condition_id"]:
        raise ValueError("sim and real condition IDs differ")

    report: dict[str, object] = {
        "schema": "go2_sim_real_pair_qa_v2",
        "status": "pass",
        "condition_id": real["condition_id"],
        "real_source": real["source_dataset"],
        "sim_source": sim["source_dataset"],
        "groups": {},
    }
    for name, (key, start, end) in GROUPS.items():
        real_values = real[key][:, start:end].numpy()
        sim_values = sim[key][:, start:end].numpy()
        metrics = dimension_metrics(real_values, sim_values)
        report["groups"][name] = {
            **metrics,
            "wasserstein_mean": float(np.mean(metrics["wasserstein"])),
            "wasserstein_max": float(np.max(metrics["wasserstein"])),
            "ks_mean": float(np.mean(metrics["ks"])),
            "ks_max": float(np.max(metrics["ks"])),
            "real": summary(real_values),
            "sim": summary(sim_values),
        }

    real_actions = real["actions"].numpy()
    sim_actions = sim["actions"].numpy()
    real_contacts = real["contacts"].numpy()
    sim_contacts = sim["contacts"].numpy()
    report["action_saturation_fraction"] = {
        "real": float(np.mean(np.abs(real_actions) >= 0.999)),
        "sim": float(np.mean(np.abs(sim_actions) >= 0.999)),
    }
    report["contact_fraction_per_foot"] = {
        "real": real_contacts.mean(axis=0).tolist(),
        "sim": sim_contacts.mean(axis=0).tolist(),
    }
    report["all_four_contact_fraction"] = {
        "real": float(np.mean(real_contacts.sum(axis=1) == 4)),
        "sim": float(np.mean(sim_contacts.sum(axis=1) == 4)),
    }
    report["termination_count"] = {
        "real": int((real["terminations"] > 0.5).sum()),
        "sim": int((sim["terminations"] > 0.5).sum()),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "condition_id": report["condition_id"],
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
