#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ROOT="${ROOT_OVERRIDE:-${REPO}/logs/rwm_datasets_v5/sim_exact_40_20_20_10_10}"
PARTS="${ROOT}/parts"
mkdir -p "${PARTS}"
cd "${REPO}"
PY="${REPO}/.venv/bin/python"

select_part() {
  local gap="$1"; shift
  [[ -s "${ROOT}/raw/${gap}.pt" ]] || { echo "Missing raw pool ${gap}" >&2; exit 2; }
  [[ ! -e "${PARTS}/${gap}.pt" ]] || { echo "Refusing to overwrite ${PARTS}/${gap}.pt" >&2; exit 2; }
  "${PY}" scripts/reinforcement_learning/rwm_dataset/select_go2_deterministic_segments_v4.py \
    --input "${ROOT}/raw/${gap}.pt" --output "${PARTS}/${gap}.pt" \
    --report "${PARTS}/${gap}.selection.json" --condition-id "${gap}" \
    --target-counts "$@" --quota-tolerance-fraction 0.0 \
    --minimum-interval 40 --maximum-interval 400 --seed 42 --expert-only
}

select_part g0   800 2500 1000 800 1400 1700 500 1300
select_part rr05 400 1250 500 400 700 850 250 650
select_part p5   400 1250 500 400 700 850 250 650
select_part rr03 200 625 250 200 350 425 125 325
select_part p75  200 625 250 200 350 425 125 325

"${PY}" scripts/reinforcement_learning/rwm_dataset/merge_go2_condition_datasets_v4.py \
  --input g0="${PARTS}/g0.pt" --expected-count g0=10000 \
  --input rr05="${PARTS}/rr05.pt" --expected-count rr05=5000 \
  --input p5="${PARTS}/p5.pt" --expected-count p5=5000 \
  --input rr03="${PARTS}/rr03.pt" --expected-count rr03=2500 \
  --input p75="${PARTS}/p75.pt" --expected-count p75=2500 \
  --output "${ROOT}/dataset_pooled25k.pt" --report "${ROOT}/merge_report.json"

"${PY}" scripts/reinforcement_learning/rwm_dataset/audit_go2_sequence_quality_v4.py \
  --input "${ROOT}/dataset_pooled25k.pt" --output-json "${ROOT}/sequence_audit.json" \
  --sequence-lengths 40 100 200
sha256sum "${ROOT}/dataset_pooled25k.pt" > "${ROOT}/dataset_pooled25k.sha256"
