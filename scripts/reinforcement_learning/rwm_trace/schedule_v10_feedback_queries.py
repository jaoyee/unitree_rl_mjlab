"""Choose a bounded, partition/mode-balanced feedback query set from a V10 pair pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.v10_protocol import COMMAND_MODES


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _bucket(row: dict) -> tuple[str, str]:
    partition = str(row.get("split_partition", ""))
    left = str((row.get("trajectory_i") or {}).get("command_mode", ""))
    right = str((row.get("trajectory_j") or {}).get("command_mode", ""))
    if partition not in {"train", "val"} or left != right or left not in COMMAND_MODES:
        raise ValueError(f"Invalid V10 pair bucket for {row.get('pair_id')}: {partition}/{left}/{right}")
    return partition, left


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--minimum-per-partition-mode", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    rows = _read_jsonl(Path(args.pool))
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_bucket(row)].append(row)
    required = [(partition, mode) for partition in ("train", "val") for mode in COMMAND_MODES]
    minimum_total = len(required) * args.minimum_per_partition_mode
    if args.budget < minimum_total:
        raise ValueError(f"Query budget {args.budget} is below the V10 minimum {minimum_total}.")
    for key in required:
        buckets[key].sort(
            key=lambda row: hashlib.sha256(
                f"{args.seed}:{row['pair_id']}".encode("utf-8")
            ).hexdigest()
        )
        if len(buckets[key]) < args.minimum_per_partition_mode:
            raise ValueError(f"V10 pair pool lacks {key}: {len(buckets[key])}")

    selected: list[dict] = []
    offsets: dict[tuple[str, str], int] = {}
    for key in required:
        selected.extend(buckets[key][: args.minimum_per_partition_mode])
        offsets[key] = args.minimum_per_partition_mode
    while len(selected) < args.budget:
        progressed = False
        for key in required:
            offset = offsets[key]
            if offset < len(buckets[key]) and len(selected) < args.budget:
                selected.append(buckets[key][offset])
                offsets[key] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != args.budget:
        raise ValueError(f"V10 query scheduler produced {len(selected)}/{args.budget} pairs.")
    selected.sort(key=lambda row: str(row["pair_id"]))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".new")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "schema": "go2_trace_v10_feedback_schedule_v1",
        "pool_size": len(rows),
        "query_count": len(selected),
        "minimum_per_partition_mode": args.minimum_per_partition_mode,
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
