#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PAYLOAD_ROOT="${PAYLOAD_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/payload_20260713_payload_0_vs_5to10kg}"
RR_ROOT="${RR_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/rr_calf_20260713_rrcalf_1_vs_0p5to1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

finalize_one() {
  local stage="$1" preset="$2" scale="$3" payload_min="$4" payload_max="$5"
  local strength_min="$6" strength_max="$7"
  if STAGE_DIR="${stage}" OUTPUT_DIR="${stage}/run" \
    DR_PRESET="${preset}" DR_SCALE="${scale}" \
    PAYLOAD_MASS_MIN_KG="${payload_min}" PAYLOAD_MASS_MAX_KG="${payload_max}" \
    RR_CALF_STRENGTH_MIN="${strength_min}" RR_CALF_STRENGTH_MAX="${strength_max}" \
    bash "${SCRIPT_DIR}/finalize_normal_expert.sh"; then
    return 0
  fi
  local code=$?
  [[ "${code}" -eq 3 ]] && return 0
  return "${code}"
}

while true; do
  finalize_one "${PAYLOAD_ROOT}/G0_D0" calibrated_default 1.0 0.0 0.0 1.0 1.0
  finalize_one "${PAYLOAD_ROOT}/G0_D1" calibrated_friend_flat 0.5 0.0 0.0 1.0 1.0
  finalize_one "${PAYLOAD_ROOT}/G0_D2" calibrated_friend_flat 1.0 0.0 0.0 1.0 1.0
  finalize_one "${PAYLOAD_ROOT}/G1_D0" calibrated_default 1.0 5.0 10.0 1.0 1.0
  finalize_one "${PAYLOAD_ROOT}/G1_D1" calibrated_friend_flat 0.5 5.0 10.0 1.0 1.0
  finalize_one "${PAYLOAD_ROOT}/G1_D2" calibrated_friend_flat 1.0 5.0 10.0 1.0 1.0
  finalize_one "${RR_ROOT}/G1_D0" calibrated_default 1.0 0.0 0.0 0.5 1.0
  finalize_one "${RR_ROOT}/G1_D1" calibrated_friend_flat 0.5 0.0 0.0 0.5 1.0
  finalize_one "${RR_ROOT}/G1_D2" calibrated_friend_flat 1.0 0.0 0.0 0.5 1.0

  completed=0
  for stage in \
    "${PAYLOAD_ROOT}"/G{0,1}_D{0,1,2} \
    "${RR_ROOT}"/G1_D{0,1,2}; do
    if [[ -f "${stage}/summary.json" ]] \
      && grep -q '"status": "completed"' "${stage}/summary.json"; then
      completed=$((completed + 1))
    fi
  done
  printf '[%s] finalized %d/9 unique experts\n' "$(date -Is)" "${completed}"
  [[ "${completed}" -eq 9 ]] && exit 0
  sleep "${POLL_SECONDS}"
done
