"""Derive one feasible-looking signed magnitude target from five real audits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


AXES = ("x", "y", "yaw")
BUCKETS = ("neg_b0", "neg_b1", "neg_b2", "pos_b0", "pos_b1", "pos_b2")
AXIS_TOTALS = {"x": 17250, "y": 10500, "yaw": 10750}


def rounded_allocation(weights: list[float], total: int) -> list[int]:
    scaled = [weight * total / sum(weights) for weight in weights]
    result = [math.floor(value) for value in scaled]
    for index in sorted(range(len(weights)), key=lambda i: scaled[i] - result[i], reverse=True)[: total - sum(result)]:
        result[index] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    datasets = audit["datasets"]
    output_counts = []
    diagnostics = {}
    for axis in AXES:
        common = {
            bucket: min(dataset["marginal_axis_counts"][axis].get(bucket, 0) for dataset in datasets.values())
            for bucket in BUCKETS
        }
        # Equalize positive/negative capacity within each magnitude bin before allocation.
        symmetric = []
        for bucket in BUCKETS:
            sign, magnitude = bucket.split("_")
            paired = ("pos" if sign == "neg" else "neg") + "_" + magnitude
            symmetric.append(float(min(common[bucket], common[paired])))
        target = rounded_allocation(symmetric, AXIS_TOTALS[axis])
        output_counts.extend(target)
        diagnostics[axis] = {
            "bucket_order": BUCKETS,
            "common_available_counts": common,
            "symmetric_capacity_weights": symmetric,
            "target_counts": target,
            "target_total": sum(target),
            "target_is_soft": True,
            "symmetric_common_capacity_total": sum(symmetric),
        }
    payload = {
        "schema": "go2_common_signed_magnitude_target_v2",
        "source_audit": str(Path(args.audit).resolve()),
        "axis_order": AXES,
        "bucket_order_per_axis": BUCKETS,
        "marginal_target_counts": output_counts,
        "diagnostics": diagnostics,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
