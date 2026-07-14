#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PAYLOAD_ROOT="${PAYLOAD_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/payload_20260713_payload_0_vs_5to10kg}"
RR_ROOT="${RR_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/rr_calf_20260713_rrcalf_1_vs_0p5to1}"

for dr in D0 D1 D2; do
  source_stage="${PAYLOAD_ROOT}/G0_${dr}"
  target_stage="${RR_ROOT}/G0_${dr}"
  while ! [[ -f "${source_stage}/summary.json" ]] \
    || ! grep -q '"status": "completed"' "${source_stage}/summary.json"; do
    sleep 60
  done
  mkdir -p "${target_stage}"
  cp "${source_stage}/summary.json" "${target_stage}/summary.json"
  cp "${source_stage}/artifact_path.txt" "${target_stage}/artifact_path.txt"
  cp "${source_stage}/artifact_sha256.txt" "${target_stage}/artifact_sha256.txt"
  printf '%s\n' "${source_stage}" > "${target_stage}/shared_baseline_source.txt"
  printf '[%s] aliased RR G0_%s to %s\n' "$(date -Is)" "${dr}" "${source_stage}"
done
