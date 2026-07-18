#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

for condition in g0 rr05 rr03 p5 p75; do
  output_dir="logs/rwm_datasets_v2/real_v2/${condition}"
  .venv/bin/python \
    scripts/reinforcement_learning/rwm_dataset/build_go2_termination_aux_v2.py \
    --input "$output_dir/threshold_18/dataset_all.pt" \
    --output "$output_dir/termination_aux.pt" \
    --report "$output_dir/termination_aux_report.json" \
    --history 40 \
    --condition-id "$condition" \
    >"$output_dir/termination_aux.log" 2>&1
done

date --iso-8601=seconds > logs/rwm_datasets_v2/real_v2/TERMINATION_AUX_COMPLETE
echo "All real V2 termination auxiliary reports completed."
