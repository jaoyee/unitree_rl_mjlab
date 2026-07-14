"""Select a conservative friend-flat DR combination from short diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DATA_PATTERNS = {
    "transitions": re.compile(r"^\s*transitions:\s*(\d+)", re.MULTILINE),
    "mean_reward": re.compile(r"^\s*mean_reward:\s*([-+0-9.eE]+)", re.MULTILINE),
    "mean_episode_length": re.compile(
        r"^\s*mean_episode_length:\s*([-+0-9.eE]+)", re.MULTILINE
    ),
    "termination_count": re.compile(r"^\s*termination_count:\s*(\d+)", re.MULTILINE),
}

METRIC_KEYS = {
    "eval_state_loss": "Model/eval_state_loss",
    "mse_1": "Model/1_step_mse",
    "mse_8": "Model/8_step_mse",
    "mse_32": "Model/32_step_mse",
    "mse_100": "Model/100_step_mse",
    "trajectory_error": "Model/traj_autoregressive_error",
    "epistemic_uncertainty": "Model/epistemic_uncertainty_mean",
    "contact_accuracy": "Model/contact_accuracy",
    "termination_accuracy": "Model/termination_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--baseline_name", default="nominal")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=COMPONENTS",
        help="Repeatable candidate mapping, e.g. core_phys=friction,mass_com,motor,push.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reward_ratio_min", type=float, default=0.80)
    parser.add_argument("--episode_length_ratio_min", type=float, default=0.85)
    parser.add_argument("--termination_ratio_max", type=float, default=1.50)
    parser.add_argument("--trajectory_error_ratio_max", type=float, default=1.30)
    parser.add_argument("--mse_100_ratio_max", type=float, default=2.00)
    parser.add_argument("--epistemic_ratio_max", type=float, default=1.40)
    parser.add_argument("--contact_accuracy_ratio_min", type=float, default=0.95)
    parser.add_argument("--randomization_scale", type=float, default=1.0)
    return parser.parse_args()


def parse_candidates(values: list[str]) -> dict[str, tuple[str, ...]]:
    if not values:
        values = [
            "core_phys=friction,mass_com,motor,push",
            "core_phys_obs=friction,mass_com,motor,push,observation",
        ]
    parsed: dict[str, tuple[str, ...]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid candidate {value!r}; expected NAME=COMPONENTS.")
        name, raw_components = value.split("=", maxsplit=1)
        name = name.strip()
        components = tuple(
            component.strip()
            for component in raw_components.replace(" ", ",").split(",")
            if component.strip()
        )
        if not name or not components:
            raise ValueError(f"Invalid candidate {value!r}.")
        parsed[name] = components
    return parsed


def latest_file(root: Path, name: str) -> Path:
    paths = list(root.rglob(name))
    if not paths:
        raise FileNotFoundError(f"No {name} found under {root}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def load_record(root: Path, name: str, components: tuple[str, ...]) -> dict[str, Any]:
    component_root = root / name
    state_path = component_root / "state.txt"
    state_text = state_path.read_text(encoding="utf-8") if state_path.exists() else ""
    status_matches = re.findall(r"^status=(.+)$", state_text, flags=re.MULTILINE)
    status = status_matches[-1].strip() if status_matches else "missing"

    collect_log = component_root / "01_collect_dataset.log"
    collect_text = collect_log.read_text(encoding="utf-8", errors="replace")
    data: dict[str, Any] = {}
    for key, pattern in DATA_PATTERNS.items():
        match = pattern.search(collect_text)
        if match is None:
            raise ValueError(f"Missing {key} in {collect_log}")
        data[key] = int(match.group(1)) if key in {"transitions", "termination_count"} else float(match.group(1))

    metrics_path = latest_file(component_root / "world_model", "final_metrics.json")
    raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = {name: float(raw_metrics[key]) for name, key in METRIC_KEYS.items()}
    return {
        "name": name,
        "components": list(components),
        "status": status,
        "component_root": str(component_root.resolve()),
        "collect_log": str(collect_log.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        **data,
        **metrics,
    }


def finite_record(record: dict[str, Any]) -> bool:
    numeric_values = [
        value
        for value in record.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return bool(numeric_values) and all(math.isfinite(float(value)) for value in numeric_values)


def evaluate_candidate(
    record: dict[str, Any],
    baseline: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, bool], float]:
    termination_limit = max(1.0, baseline["termination_count"] * args.termination_ratio_max)
    gates = {
        "completed": record["status"] == "completed",
        "finite": finite_record(record),
        "transitions": record["transitions"] >= baseline["transitions"],
        "reward": record["mean_reward"] >= baseline["mean_reward"] * args.reward_ratio_min,
        "episode_length": record["mean_episode_length"]
        >= baseline["mean_episode_length"] * args.episode_length_ratio_min,
        "terminations": record["termination_count"] <= termination_limit,
        "trajectory_error": record["trajectory_error"]
        <= baseline["trajectory_error"] * args.trajectory_error_ratio_max,
        "mse_100": record["mse_100"] <= baseline["mse_100"] * args.mse_100_ratio_max,
        "epistemic_uncertainty": record["epistemic_uncertainty"]
        <= baseline["epistemic_uncertainty"] * args.epistemic_ratio_max,
        "contact_accuracy": record["contact_accuracy"]
        >= baseline["contact_accuracy"] * args.contact_accuracy_ratio_min,
    }
    score_terms = (
        baseline["mean_reward"] / max(record["mean_reward"], 1e-8),
        baseline["mean_episode_length"] / max(record["mean_episode_length"], 1e-8),
        record["trajectory_error"] / max(baseline["trajectory_error"], 1e-8),
        record["mse_100"] / max(baseline["mse_100"], 1e-8),
        record["epistemic_uncertainty"] / max(baseline["epistemic_uncertainty"], 1e-8),
        baseline["contact_accuracy"] / max(record["contact_accuracy"], 1e-8),
    )
    return gates, float(sum(score_terms) / len(score_terms))


def write_markdown(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Friend-flat DR combination selection",
        "",
        f"Status: `{result['status']}`",
        f"Selected candidate: `{result.get('selected_name') or 'none'}`",
        f"Selected components: `{','.join(result.get('selected_components') or [])}`",
        f"Selected randomization scale: `{result.get('selected_randomization_scale')}`",
        "",
        "| Candidate | Components | Reward | Episode length | Terminations | Trajectory error | 100-step MSE | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["candidates"]:
        lines.append(
            "| {name} | {components_label} | {mean_reward:.4f} | {mean_episode_length:.2f} | "
            "{termination_count} | {trajectory_error:.4f} | {mse_100:.4f} | {passed_label} |".format(
                **record,
                components_label=",".join(record["components"]),
                passed_label="yes" if record["passed"] else "no",
            )
        )
    lines.extend(["", "## Gates", ""])
    for record in result["candidates"]:
        failed = [name for name, passed in record["gates"].items() if not passed]
        lines.append(f"- `{record['name']}`: " + ("passed" if not failed else f"failed {', '.join(failed)}"))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidates = parse_candidates(args.candidate)
    baseline = load_record(args.baseline_root, args.baseline_name, ())
    if baseline["status"] != "completed" or not finite_record(baseline):
        raise RuntimeError("Nominal baseline is incomplete or non-finite.")

    records: list[dict[str, Any]] = []
    for name, components in candidates.items():
        record = load_record(args.candidate_root, name, components)
        gates, score = evaluate_candidate(record, baseline, args)
        record["gates"] = gates
        record["score"] = score
        record["passed"] = all(gates.values())
        records.append(record)

    passing = [record for record in records if record["passed"]]
    selected = min(
        passing,
        key=lambda record: (-len(record["components"]), record["score"], record["name"]),
        default=None,
    )
    result = {
        "status": "selected" if selected is not None else "no_candidate_passed",
        "selection_rule": "largest component set passing all preregistered gates, then lowest normalized score",
        "baseline": baseline,
        "thresholds": {
            "reward_ratio_min": args.reward_ratio_min,
            "episode_length_ratio_min": args.episode_length_ratio_min,
            "termination_ratio_max": args.termination_ratio_max,
            "trajectory_error_ratio_max": args.trajectory_error_ratio_max,
            "mse_100_ratio_max": args.mse_100_ratio_max,
            "epistemic_ratio_max": args.epistemic_ratio_max,
            "contact_accuracy_ratio_min": args.contact_accuracy_ratio_min,
        },
        "candidates": records,
        "selected_name": selected["name"] if selected else None,
        "selected_components": selected["components"] if selected else [],
        "selected_randomization_scale": args.randomization_scale if selected else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "selection.env").write_text(
        f"SELECTION_STATUS={result['status']}\n"
        f"SELECTED_NAME={result['selected_name'] or ''}\n"
        f"SELECTED_COMPONENTS={','.join(result['selected_components'])}\n"
        f"SELECTED_RANDOMIZATION_SCALE={result['selected_randomization_scale'] or ''}\n",
        encoding="utf-8",
    )
    write_markdown(result, args.output_dir / "selection.md")
    print(json.dumps({
        "status": result["status"],
        "selected_name": result["selected_name"],
        "selected_components": result["selected_components"],
        "selected_randomization_scale": result["selected_randomization_scale"],
    }, indent=2))


if __name__ == "__main__":
    main()
