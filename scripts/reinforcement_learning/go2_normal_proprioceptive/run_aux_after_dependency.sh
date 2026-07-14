#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?gpu id}"
DEPENDENCY_RELATIVE_DIR="${2:?dependency relative directory}"
JOB_ID="${3:?job id}"
JOB_RELATIVE_DIR="${4:?job relative directory}"

REPO_ROOT="${REPO_ROOT:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712}"
DEPENDENCY_STATE="${EXPERIMENT_ROOT}/${DEPENDENCY_RELATIVE_DIR}/state.txt"

while true; do
  status="$(awk -F= '$1 == "status" {print $2; exit}' "${DEPENDENCY_STATE}")"
  case "${status}" in
    completed) break ;;
    failed|blocked) exit 1 ;;
  esac
  sleep 15
done

exec "${REPO_ROOT}/scripts/reinforcement_learning/go2_normal_proprioceptive/run_aux_job.sh" \
  "${GPU_ID}" "${JOB_ID}" "${JOB_RELATIVE_DIR}"
