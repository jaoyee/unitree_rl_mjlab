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

mkdir -p "logs/rwm_datasets_v2/qa_features/$kind"
for condition in g0 rr05 rr03 p5 p75; do
  .venv/bin/python \
    scripts/reinforcement_learning/rwm_dataset/export_go2_v2_qa_features.py \
    --dataset "$base/$condition/dataset.pt" \
    --output "logs/rwm_datasets_v2/qa_features/$kind/$condition.pt" \
    --condition-id "$condition" \
    --domain "$kind"
done

date --iso-8601=seconds > "logs/rwm_datasets_v2/qa_features/$kind/EXPORT_COMPLETE"
