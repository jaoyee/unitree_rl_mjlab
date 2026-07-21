#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


CODE_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/home/xjy/writing/CORL_2026/outputs/trace_feedback_score/labels")
    parser.add_argument("--schema", type=str, default=str(CODE_ROOT / "scripts/schemas/feedback_label_batch.schema.json"))
    parser.add_argument("--repo-root", type=str, default=str(CODE_ROOT))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--minimum-filtered-labels",
        type=int,
        default=10,
        help="Retry only the weakest batches until this many unique labels pass the confidence gate.",
    )
    parser.add_argument(
        "--max-low-confidence-relabels",
        type=int,
        default=7,
        help="Maximum additional low-confidence batch queries per output directory.",
    )
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--lock-path", type=str, default=None)
    parser.add_argument("--codex-model", type=str, default=None)
    parser.add_argument(
        "--codex-service-tier",
        type=str,
        default="default",
        help="Service tier override for codex exec calls; keeps user config unchanged.",
    )
    return parser.parse_args()


def iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_bytes(payload.encode("utf-8"))


def build_cache_identity(args, prompts: List[dict]) -> dict:
    pair_fingerprints = []
    for row in prompts:
        pair_fingerprints.append(
            {
                "pair_id": str(row["pair_id"]),
                "pair_sha256": _canonical_sha256(
                    {
                        "prompt": row.get("prompt"),
                        "trajectory_i": row.get("trajectory_i", row.get("traj_i")),
                        "trajectory_j": row.get("trajectory_j", row.get("traj_j")),
                    }
                ),
            }
        )
    schema_path = Path(args.schema).expanduser().resolve()
    return {
        "schema_version": 1,
        "prompts_path": str(Path(args.prompts).expanduser().resolve()),
        "prompts_file_sha256": _sha256_path(Path(args.prompts).expanduser().resolve()),
        "effective_pairs_sha256": _canonical_sha256(pair_fingerprints),
        "schema_sha256": _sha256_path(schema_path),
        "batch_size": int(args.batch_size),
        "max_pairs": args.max_pairs,
        "pair_fingerprints": pair_fingerprints,
    }


def bind_output_cache(out: Path, identity: dict) -> None:
    identity_path = out / "label_cache_identity.json"
    cached_responses = list((out / "raw_batches").glob("**/*_response.json"))
    if identity_path.exists():
        existing = load_json(identity_path)
        if existing != identity:
            raise RuntimeError(
                "Label cache identity mismatch. Refusing to bind cached Codex responses to "
                "different pairs/prompts; use a new output directory."
            )
        return
    if cached_responses:
        raise RuntimeError(
            "Legacy label responses exist without label_cache_identity.json. Refusing unsafe "
            "reuse; use a new output directory."
        )
    temporary = identity_path.with_suffix(identity_path.suffix + ".new")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(identity_path)


def build_batch_prompt(rows: List[dict]) -> str:
    examples = []
    for row in rows:
        examples.append({
            "pair_id": row["pair_id"],
            "prompt": row["prompt"],
        })
    return (
        "You are labeling robot trajectory pairs for offline robot learning.\n"
        "For each pair, read the structured diagnostics and return one label.\n"
        "Do not use behavior_return or simulator_return as the sole criterion; return may be misleading under imperfect simulator mismatch.\n"
        "For conflict_return_vs_motion pairs, explicitly decide whether smoother, less saturated, more plausible motion should override higher return.\n"
        "Set return_used_as_primary=true only when return was the dominant reason. Set motion_quality_overrode_return=true when you pick the lower-return trajectory because its motion diagnostics are better.\n"
        "Return strict JSON matching the provided schema, with exactly one entry per pair_id.\n\n"
        + json.dumps({"pairs": examples}, indent=2, sort_keys=True)
    )


def call_codex(prompt_path: Path, response_path: Path, stdout_path: Path, args) -> bool:
    cmd = [
        "codex",
        "exec",
        "--cd",
        str(args.repo_root),
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(args.schema),
        "-o",
        str(response_path),
        "-",
    ]
    if args.codex_model:
        cmd[2:2] = ["--model", str(args.codex_model)]
    if args.codex_service_tier:
        cmd[2:2] = ["-c", f'service_tier="{args.codex_service_tier}"']
    prompt_text = prompt_path.read_text(encoding="utf-8")
    try:
        proc = subprocess.run(
            cmd,
            input=prompt_text,
            text=True,
            capture_output=True,
            timeout=float(args.timeout_sec),
        )
        stdout_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        return proc.returncode == 0 and response_path.exists()
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        stdout_path.write_text(
            f"{stdout}\n{stderr}\nTIMEOUT after {args.timeout_sec} seconds\n",
            encoding="utf-8",
        )
        return False


def call_codex_with_optional_lock(prompt_path: Path, response_path: Path, stdout_path: Path, args) -> bool:
    if not args.lock_path:
        return call_codex(prompt_path, response_path, stdout_path, args)
    lock_path = Path(args.lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return call_codex(prompt_path, response_path, stdout_path, args)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def parse_response(response_path: Path) -> dict:
    text = response_path.read_text(encoding="utf-8").strip()
    return json.loads(text)


def _label_rank(label: dict) -> tuple[int, float]:
    complete = int(
        label.get("feedback") in {"i", "j"}
        and bool(str(label.get("reason", "")).strip())
    )
    try:
        confidence = float(label.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return complete, confidence


def merge_labels(prompt_rows: List[dict], label_rows: List[dict], confidence_threshold: float):
    prompt_by_id: Dict[str, dict] = {row["pair_id"]: row for row in prompt_rows}
    best_by_id: Dict[str, dict] = {}
    for label in label_rows:
        pair_id = str(label.get("pair_id", ""))
        if pair_id not in prompt_by_id:
            continue
        previous = best_by_id.get(pair_id)
        if previous is None or _label_rank(label) > _label_rank(previous):
            best_by_id[pair_id] = label
    raw = []
    filtered = []
    for prompt_row in prompt_rows:
        pair_id = str(prompt_row["pair_id"])
        label = best_by_id.get(pair_id)
        if label is None:
            continue
        prompt = prompt_by_id.get(pair_id)
        merged = {**label}
        if prompt:
            for key in (
                "trajectory_source",
                "task",
                "seed",
                "offline_data_state_id",
                "split_group_id",
                "dataset_state_pos",
                "pair_mode",
                "pair_type",
                "same_episode",
                "same_state_pair",
                "traj_i",
                "traj_j",
                "traj_i_summary_path",
                "traj_j_summary_path",
                "traj_i_npz_path",
                "traj_j_npz_path",
                "episode_i",
                "episode_j",
                "window_start_i",
                "window_start_j",
                "traj_i_return",
                "traj_j_return",
                "traj_i_motion_quality_proxy",
                "traj_j_motion_quality_proxy",
                "higher_return_side",
                "higher_motion_quality_side",
            ):
                if key in prompt:
                    merged[key] = prompt[key]
        merged.setdefault("return_used_as_primary", False)
        merged.setdefault("motion_quality_overrode_return", False)
        raw.append(merged)
        keep = (
            pair_id
            and prompt is not None
            and merged.get("feedback") in {"i", "j"}
            and float(merged.get("confidence", 0.0)) >= float(confidence_threshold)
            and bool(str(merged.get("reason", "")).strip())
        )
        if keep:
            filtered.append(merged)
    return raw, filtered


def response_labels(response_path: Path) -> List[dict]:
    try:
        parsed = parse_response(response_path)
    except Exception:
        return []
    labels = parsed.get("labels", [])
    return labels if isinstance(labels, list) else []


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    raw_dir = out / "raw_batches"
    relabel_dir = raw_dir / "low_confidence_relabels"
    batch_prompt_dir = out / "batch_prompts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    relabel_dir.mkdir(parents=True, exist_ok=True)
    batch_prompt_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(iter_jsonl(args.prompts))
    if args.max_pairs is not None:
        prompts = prompts[: args.max_pairs]
    bind_output_cache(out, build_cache_identity(args, prompts))
    minimum_filtered = min(max(int(args.minimum_filtered_labels), 0), len(prompts))

    if shutil.which("codex") is None:
        write_jsonl(out / "manual_label_todo.jsonl", prompts)
        message = "codex command not found; wrote manual_label_todo.jsonl"
        print(message)
        if minimum_filtered > 0:
            raise SystemExit(message)
        return

    all_labels = []
    failed = []
    for start in range(0, len(prompts), args.batch_size):
        batch_idx = start // args.batch_size
        batch = prompts[start: start + args.batch_size]
        prompt_path = batch_prompt_dir / f"batch_{batch_idx:04d}_prompt.txt"
        response_path = raw_dir / f"batch_{batch_idx:04d}_response.json"
        stdout_path = raw_dir / f"batch_{batch_idx:04d}_stdout.jsonl"
        prompt_path.write_text(build_batch_prompt(batch), encoding="utf-8")

        parsed = None
        if response_path.exists():
            try:
                parsed = parse_response(response_path)
            except Exception:
                parsed = None
        if parsed is None:
            for attempt in range(args.max_retries + 1):
                if call_codex_with_optional_lock(prompt_path, response_path, stdout_path, args):
                    try:
                        parsed = parse_response(response_path)
                        break
                    except Exception as exc:
                        stdout_path.write_text(stdout_path.read_text(encoding="utf-8") + f"\nPARSE_ERROR: {exc}\n", encoding="utf-8")
                if attempt == args.max_retries:
                    failed.append(batch_idx)
        if parsed is not None:
            labels = parsed.get("labels", [])
            all_labels.extend(labels)

    # Existing low-confidence relabels are additive.  For every pair,
    # merge_labels keeps the complete judgment with the highest confidence,
    # so a retry can never discard a previously accepted label.
    for response_path in sorted(relabel_dir.glob("relabel_*_batch_*_response.json")):
        all_labels.extend(response_labels(response_path))

    raw, filtered = merge_labels(prompts, all_labels, args.confidence_threshold)
    existing_attempts = len(list(relabel_dir.glob("relabel_*_batch_*_stdout.jsonl")))
    while (
        len(filtered) < minimum_filtered
        and existing_attempts < max(int(args.max_low_confidence_relabels), 0)
    ):
        filtered_ids = {str(row.get("pair_id", "")) for row in filtered}
        attempts_by_batch: Dict[int, int] = {}
        for stdout_path in relabel_dir.glob("relabel_*_batch_*_stdout.jsonl"):
            try:
                batch_idx = int(stdout_path.name.split("_batch_", 1)[1].split("_", 1)[0])
            except (IndexError, ValueError):
                continue
            attempts_by_batch[batch_idx] = attempts_by_batch.get(batch_idx, 0) + 1
        batch_candidates = []
        for start in range(0, len(prompts), args.batch_size):
            batch_idx = start // args.batch_size
            batch = prompts[start: start + args.batch_size]
            accepted = sum(str(row["pair_id"]) in filtered_ids for row in batch)
            missing = len(batch) - accepted
            batch_candidates.append(
                (attempts_by_batch.get(batch_idx, 0), -missing, batch_idx, batch)
            )
        _, _, batch_idx, batch = min(batch_candidates, key=lambda row: row[:3])
        existing_attempts += 1
        stem = f"relabel_{existing_attempts:04d}_batch_{batch_idx:04d}"
        prompt_path = relabel_dir / f"{stem}_prompt.txt"
        response_path = relabel_dir / f"{stem}_response.json"
        stdout_path = relabel_dir / f"{stem}_stdout.jsonl"
        prompt_path.write_text(build_batch_prompt(batch), encoding="utf-8")
        print(
            f"Filtered-label gate {len(filtered)}/{minimum_filtered}; "
            f"relabeling low-confidence batch {batch_idx} "
            f"({existing_attempts}/{args.max_low_confidence_relabels})"
        )
        parsed = None
        for attempt in range(args.max_retries + 1):
            if call_codex_with_optional_lock(prompt_path, response_path, stdout_path, args):
                try:
                    parsed = parse_response(response_path)
                    break
                except Exception as exc:
                    stdout_path.write_text(
                        stdout_path.read_text(encoding="utf-8") + f"\nPARSE_ERROR: {exc}\n",
                        encoding="utf-8",
                    )
            if attempt == args.max_retries:
                failed.append(f"low_confidence_relabel_{existing_attempts}")
        if parsed is not None:
            all_labels.extend(parsed.get("labels", []))
        raw, filtered = merge_labels(prompts, all_labels, args.confidence_threshold)

    if failed:
        (out / "failed_batches.txt").write_text("\n".join(str(x) for x in failed), encoding="utf-8")

    write_jsonl(out / "codex_labels_raw.jsonl", raw)
    write_jsonl(out / "codex_labels_filtered.jsonl", filtered)
    conflict_raw = [row for row in raw if row.get("pair_type") == "conflict_return_vs_motion"]
    conflict_filtered = [row for row in filtered if row.get("pair_type") == "conflict_return_vs_motion"]
    feedback_counts = {}
    for row in raw:
        feedback_value = row.get("feedback", "missing")
        feedback_counts[feedback_value] = int(feedback_counts.get(feedback_value, 0) + 1)
    confidence_values = []
    for row in raw:
        try:
            confidence_values.append(float(row.get("confidence", 0.0)))
        except (TypeError, ValueError):
            pass
    return_primary_count = sum(1 for row in filtered if bool(row.get("return_used_as_primary", False)))
    motion_override_count = sum(1 for row in filtered if bool(row.get("motion_quality_overrode_return", False)))
    summary = {
        "num_prompt_pairs": len(prompts),
        "num_raw_labels": len(raw),
        "num_filtered_labels": len(filtered),
        "confidence_threshold": float(args.confidence_threshold),
        "failed_batches": failed,
        "feedback_counts": feedback_counts,
        "return_used_as_primary_count": int(return_primary_count),
        "return_used_as_primary_ratio": float(return_primary_count / max(len(filtered), 1)),
        "motion_quality_overrode_return_count": int(motion_override_count),
        "motion_quality_overrode_return_ratio": float(motion_override_count / max(len(filtered), 1)),
        "conflict_raw_labels": int(len(conflict_raw)),
        "conflict_filtered_labels": int(len(conflict_filtered)),
        "conflict_pair_filtered_ratio": float(len(conflict_filtered) / max(len(conflict_raw), 1)),
        "mean_confidence": float(sum(confidence_values) / max(len(confidence_values), 1)),
    }
    (out / "label_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if len(filtered) < minimum_filtered:
        raise SystemExit(
            f"Filtered-label gate failed after bounded relabeling: "
            f"{len(filtered)}/{minimum_filtered}. Use a new label directory, add pairs, "
            "or lower the explicit minimum; do not wait on the same cache forever."
        )


if __name__ == "__main__":
    main()
