#!/usr/bin/env python3
"""Create a self-describing scorer bundle without baseline-specific paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from trace_scorer.core import load_scorer, sha256_path


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.scorer).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "scorer.pt"
    source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    metadata = dict(source_checkpoint.get("metadata") or {})
    removed_path_fields = {
        key: metadata.pop(key)
        for key in ("summaries", "metric_config")
        if key in metadata
    }
    metadata["portable_export"] = {
        "source_checkpoint_sha256": sha256_path(source),
        "removed_metadata_fields": sorted(removed_path_fields),
    }
    source_checkpoint["metadata"] = metadata
    torch.save(source_checkpoint, target)
    scorer = load_scorer(target, reject_reward_features=True)
    manifest = {
        "schema": "portable_trace_scorer_bundle_v1",
        "scorer_file": "scorer.pt",
        "scorer_sha256": sha256_path(target),
        "source_scorer_sha256": sha256_path(source),
        "removed_baseline_path_metadata_fields": sorted(removed_path_fields),
        "descriptor": scorer.descriptor.to_dict(),
        "input_contract": {
            "format": "JSONL objects",
            "required_semantics": (
                "trajectory-level command response and motion-stability summaries; "
                "missing values are represented by explicit missingness indicators"
            ),
            "base_feature_names": list(scorer.base_feature_names),
        },
        "output_contract": {
            "score": "unbounded Bradley-Terry utility; higher is preferred",
            "selection": "performed outside this module and recorded in portable_trace_selection_v1",
        },
        "baseline_dependencies": {
            "rwm_checkpoint": False,
            "reward_function": False,
            "actor": False,
            "critic": False,
            "replay_builder": False,
        },
        "reward_features_rejected": True,
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
