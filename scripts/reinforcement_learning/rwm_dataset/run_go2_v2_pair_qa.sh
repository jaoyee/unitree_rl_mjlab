#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

output_dir="logs/rwm_datasets_v2/pair_qa"
mkdir -p "$output_dir"
for condition in g0 rr05 rr03 p5 p75; do
  .venv/bin/python \
    scripts/reinforcement_learning/rwm_dataset/compare_go2_v2_pair.py \
    --real "logs/rwm_datasets_v2/qa_features/real/$condition.pt" \
    --sim "logs/rwm_datasets_v2/qa_features/sim/$condition.pt" \
    --output "$output_dir/$condition.json" \
    >"$output_dir/$condition.log" 2>&1
done

date --iso-8601=seconds > "$output_dir/PAIR_QA_COMPLETE"
echo "All five sim-real V2 pair QA reports completed."
