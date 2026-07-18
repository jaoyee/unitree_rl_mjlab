#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SIDE="${SIDE:?Set SIDE to sim or real}"
case "${SIDE}" in
  sim|real) ;;
  *) echo "SIDE must be sim or real, got ${SIDE}" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-${REPO}/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO}/logs/rwm_datasets_v2/${SIDE}_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/logs/rwm_datasets_v4/${SIDE}_v4/pooled_40_20_20_10_10}"
SELECTOR="${REPO}/scripts/reinforcement_learning/rwm_dataset/select_go2_deterministic_segments_v4.py"
MERGER="${REPO}/scripts/reinforcement_learning/rwm_dataset/merge_go2_condition_datasets_v4.py"
AUDITOR="${REPO}/scripts/reinforcement_learning/rwm_dataset/audit_go2_sequence_quality_v4.py"

mkdir -p "${OUTPUT_ROOT}/parts"

select_part() {
  local name="$1" seed="$2"
  shift 2
  local input="${SOURCE_ROOT}/${name}/dataset.pt"
  local output="${OUTPUT_ROOT}/parts/${name}.pt"
  local report="${OUTPUT_ROOT}/parts/${name}.selection.json"
  [[ -f "${input}" ]] || { echo "Missing source dataset ${input}" >&2; exit 2; }
  if [[ ! -s "${output}" ]]; then
    "${PYTHON}" "${SELECTOR}" \
      --input "${input}" --output "${output}" --report "${report}" \
      --condition-id "${name}" --target-counts "$@" \
      --quota-tolerance-fraction 0.0 --minimum-interval 40 --maximum-interval 400 \
      --seed "${seed}"
  fi
}

# Overall command target is exactly:
# 2000,6250,2500,2000,3500,4250,1250,3250 (25K).
# Each condition receives the same command proportions as the pooled target.
select_part g0   420 800 2500 1000 800 1400 1700 500 1300
select_part rr05 421 400 1250  500 400  700  850 250  650
select_part p5   422 400 1250  500 400  700  850 250  650
select_part rr03 423 200  625  250 200  350  425 125  325
select_part p75  424 200  625  250 200  350  425 125  325

DATASET="${OUTPUT_ROOT}/dataset_pooled25k.pt"
REPORT="${OUTPUT_ROOT}/merge_report.json"
if [[ ! -s "${DATASET}" ]]; then
  "${PYTHON}" "${MERGER}" \
    --input "g0=${OUTPUT_ROOT}/parts/g0.pt" \
    --input "rr05=${OUTPUT_ROOT}/parts/rr05.pt" \
    --input "p5=${OUTPUT_ROOT}/parts/p5.pt" \
    --input "rr03=${OUTPUT_ROOT}/parts/rr03.pt" \
    --input "p75=${OUTPUT_ROOT}/parts/p75.pt" \
    --expected-count g0=10000 --expected-count rr05=5000 --expected-count p5=5000 \
    --expected-count rr03=2500 --expected-count p75=2500 \
    --output "${DATASET}" --report "${REPORT}"
fi

"${PYTHON}" "${AUDITOR}" --input "${DATASET}" \
  --output-json "${OUTPUT_ROOT}/sequence_audit.json" --sequence-lengths 40 100 200

sha256sum "${DATASET}" > "${OUTPUT_ROOT}/dataset_pooled25k.sha256"
printf 'side=%s\ndataset=%s\nreport=%s\naudit=%s\n' \
  "${SIDE}" "${DATASET}" "${REPORT}" "${OUTPUT_ROOT}/sequence_audit.json" \
  > "${OUTPUT_ROOT}/build_summary.txt"
