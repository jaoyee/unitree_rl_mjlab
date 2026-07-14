#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${EXPERT_POLICY_PATH:?Set EXPERT_POLICY_PATH to the best 0.5-strength expert checkpoint.}"
: "${BASELINE_ROOT:?Set BASELINE_ROOT to the completed component-diagnostic root containing nominal/.}"

RUN_ID="${RUN_ID:-friend_dr_combinations_$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-logs/queues/go2_0p5_friend_dr_combinations/${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/diagnostics/go2_0p5_friend_dr_combinations/${RUN_ID}}"
SELECTION_ROOT="${SELECTION_ROOT:-${OUTPUT_ROOT}/selection}"
GPU_IDS="${GPU_IDS:-0 1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-25}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
ORCHESTRATOR_STATE="${QUEUE_ROOT}/orchestrator_state.txt"
VALIDATION_SCALE="${VALIDATION_SCALE:-1.0}"

case "${VALIDATION_SCALE}" in
  1|1.0)
    CANDIDATE_A=core_phys
    CANDIDATE_B=core_phys_obs
    ;;
  0.5|.5)
    CANDIDATE_A=core_phys_half
    CANDIDATE_B=core_phys_half_obs
    ;;
  *)
    echo "Unsupported VALIDATION_SCALE=${VALIDATION_SCALE}; allowed: 1.0, 0.5" >&2
    exit 2
    ;;
esac

mkdir -p "${QUEUE_ROOT}" "${OUTPUT_ROOT}" "${SELECTION_ROOT}"

write_state() {
  local stage="$1" status="$2"
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "run_id=${RUN_ID}"
    echo "stage=${stage}"
    echo "status=${status}"
    echo "baseline_root=${BASELINE_ROOT}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "validation_scale=${VALIDATION_SCALE}"
  } > "${ORCHESTRATOR_STATE}"
}

on_error() {
  local exit_code=$?
  write_state "${CURRENT_STAGE:-unknown}" failed
  echo "exit_code=${exit_code}" >> "${ORCHESTRATOR_STATE}"
  exit "${exit_code}"
}
trap on_error ERR

CURRENT_STAGE=combination_diagnostics
write_state "${CURRENT_STAGE}" running
RUN_ID="${RUN_ID}" \
QUEUE_ROOT="${QUEUE_ROOT}/scheduler" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
COMPONENTS="${CANDIDATE_A} ${CANDIDATE_B}" \
GPU_IDS="${GPU_IDS}" \
MAX_PARALLEL=2 \
MIN_FREE_MIB="${MIN_FREE_MIB}" \
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT}" \
POLL_SECONDS="${POLL_SECONDS}" \
EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH}" \
bash "${SCRIPT_DIR}/launch_friend_flat_dr_component_diagnostics.sh"

CURRENT_STAGE=select_combination
write_state "${CURRENT_STAGE}" running
"${PYTHON_BIN}" "${SCRIPT_DIR}/select_friend_flat_dr_combination.py" \
  --baseline_root "${BASELINE_ROOT}" \
  --candidate_root "${OUTPUT_ROOT}" \
  --candidate "${CANDIDATE_A}=friction,mass_com,motor,push" \
  --candidate "${CANDIDATE_B}=friction,mass_com,motor,push,observation" \
  --randomization_scale "${VALIDATION_SCALE}" \
  --output_dir "${SELECTION_ROOT}" \
  2>&1 | tee "${QUEUE_ROOT}/selection.log"

selection_status="$(awk -F= '$1 == "SELECTION_STATUS" {print $2}' "${SELECTION_ROOT}/selection.env")"
if [[ "${selection_status}" == "selected" ]]; then
  write_state completed completed
else
  write_state selection_stopped no_candidate_passed
fi
