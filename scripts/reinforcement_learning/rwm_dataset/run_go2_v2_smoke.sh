#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

kind="${1:?usage: $0 real|sim}"
case "$kind" in
  real) base="logs/rwm_datasets_v2/real_v2" ;;
  sim) base="logs/rwm_datasets_v2/sim_v2" ;;
  *) echo "kind must be real or sim" >&2; exit 2 ;;
esac

for condition in g0 rr05 rr03 p5 p75; do
  output_dir="$base/$condition"
  .venv/bin/python \
    scripts/reinforcement_learning/rwm_dataset/smoke_go2_v2_dataset.py \
    --dataset "$output_dir/dataset.pt" \
    --output "$output_dir/smoke_report.json" \
    --batch-size 2 \
    >"$output_dir/smoke.log" 2>&1
done

date --iso-8601=seconds > "$base/SMOKE_COMPLETE"
echo "All $kind V2 sampler/loss smoke tests passed."
