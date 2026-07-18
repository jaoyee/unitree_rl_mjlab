#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

conditions=(g0 rr05 rr03 p5 p75)
pids=()

for condition in "${conditions[@]}"; do
  output_dir="logs/rwm_datasets_v2/sim_v2/${condition}"
  target="logs/rwm_datasets_v2/specs/${condition}_real_marginal_target_v2.json"
  mkdir -p "$output_dir"
  .venv/bin/python \
    scripts/reinforcement_learning/rwm_dataset/select_go2_balanced_transitions_v2.py \
    --input "$output_dir/expert_pool_150k.pt" \
    --output "$output_dir/dataset.pt" \
    --report "$output_dir/selection_report.json" \
    --condition-id "$condition" \
    --marginal-target-json "$target" \
    --expert-only \
    --time-limit 300 \
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
  echo "One or more sim V2 selections failed." >&2
  exit "$status"
fi

date --iso-8601=seconds > logs/rwm_datasets_v2/sim_v2/MAIN_SELECTION_COMPLETE
echo "All sim V2 main selections completed."
