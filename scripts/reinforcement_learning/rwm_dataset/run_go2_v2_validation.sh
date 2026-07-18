#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

kind="${1:?usage: $0 real|sim}"
case "$kind" in
  real)
    base="logs/rwm_datasets_v2/real_v2"
    expert_arg=()
    ;;
  sim)
    base="logs/rwm_datasets_v2/sim_v2"
    expert_arg=(--require-expert)
    ;;
  *)
    echo "kind must be real or sim" >&2
    exit 2
    ;;
esac

for condition in g0 rr05 rr03 p5 p75; do
  output_dir="$base/$condition"
  .venv/bin/python \
    scripts/reinforcement_learning/rwm_dataset/validate_go2_matched_v2.py \
    --dataset "$output_dir/dataset.pt" \
    --output "$output_dir/validation_report.json" \
    --expected-transitions 25000 \
    --minimum-valid-40-step-starts 9000 \
    "${expert_arg[@]}" \
    >"$output_dir/validation.log" 2>&1
done

date --iso-8601=seconds > "$base/VALIDATION_COMPLETE"
echo "All $kind V2 datasets passed strict validation."
