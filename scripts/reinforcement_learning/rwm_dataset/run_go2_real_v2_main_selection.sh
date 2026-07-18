#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

conditions=(g0 rr05 rr03 p5 p75)
pids=()

for condition in "${conditions[@]}"; do
  output_dir="logs/rwm_datasets_v2/real_v2/${condition}"
  mkdir -p "$output_dir"
  .venv/bin/python \
    scripts/reinforcement_learning/rwm_dataset/select_go2_real_command_coverage_dataset.py \
    --input "$output_dir/threshold_18/dataset_all.pt" \
    --output "$output_dir/dataset.pt" \
    --report-json "$output_dir/selection_report.json" \
    --condition-id "$condition" \
    >"$output_dir/main_selection.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if ((status != 0)); then
  echo "One or more real V2 selections failed." >&2
  exit "$status"
fi

date --iso-8601=seconds > logs/rwm_datasets_v2/real_v2/MAIN_SELECTION_COMPLETE
echo "All real V2 main selections completed."
