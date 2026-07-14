#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${EXPERT_POLICY_PATH:?Set EXPERT_POLICY_PATH to the best 0.5-strength expert checkpoint.}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-logs/queues/go2_0p5_friend_dr_components/${RUN_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/diagnostics/go2_0p5_friend_dr_components/${RUN_ID}}"
COMPONENTS="${COMPONENTS:-nominal friction mass_com motor delay push observation full}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
GPU_IDS="${GPU_IDS//,/ }"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-4500}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "${QUEUE_ROOT}" "${OUTPUT_ROOT}"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"
STATE_FILE="${QUEUE_ROOT}/state.txt"

read -r -a COMPONENT_LIST <<< "${COMPONENTS}"
read -r -a GPU_LIST <<< "${GPU_IDS}"
declare -A JOB_PID=()
declare -A JOB_GPU=()
declare -A GPU_STREAK=()
PICKED_GPU=""

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${QUEUE_LOG}"
}

write_state() {
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "run_id=${RUN_ID}"
    echo "status=$1"
    echo "components=${COMPONENTS}"
    echo "gpu_ids=${GPU_IDS}"
    echo "max_parallel=${MAX_PARALLEL}"
    for component in "${COMPONENT_LIST[@]}"; do
      local state="${OUTPUT_ROOT}/${component}/state.txt"
      if [[ -f "${state}" ]]; then
        echo "${component}=$(awk -F= '$1 == "status" {print $2}' "${state}" | tail -1)"
      elif [[ -n "${JOB_PID[${component}]:-}" ]]; then
        echo "${component}=running"
      else
        echo "${component}=pending"
      fi
    done
  } > "${STATE_FILE}"
}

refresh_jobs() {
  local component pid exit_code
  for component in "${!JOB_PID[@]}"; do
    pid="${JOB_PID[${component}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        exit_code=0
      else
        exit_code=$?
      fi
      log "component=${component} gpu=${JOB_GPU[${component}]} exit_code=${exit_code}"
      unset "JOB_PID[${component}]"
      unset "JOB_GPU[${component}]"
    fi
  done
}

running_count() {
  echo "${#JOB_PID[@]}"
}

gpu_in_use() {
  local gpu="$1" component
  for component in "${!JOB_GPU[@]}"; do
    [[ "${JOB_GPU[${component}]}" == "${gpu}" ]] && return 0
  done
  return 1
}

pick_gpu() {
  local query gpu free util
  PICKED_GPU=""
  query="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)"
  for gpu in "${GPU_LIST[@]}"; do
    gpu_in_use "${gpu}" && continue
    read -r free util < <(awk -F, -v target="${gpu}" '$1 + 0 == target {gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3}' <<< "${query}")
    if [[ -n "${free:-}" && "${free}" -ge "${MIN_FREE_MIB}" && "${util}" -le "${MAX_UTIL_PERCENT}" ]]; then
      GPU_STREAK[${gpu}]=$(( ${GPU_STREAK[${gpu}]:-0} + 1 ))
      if [[ "${GPU_STREAK[${gpu}]}" -ge 2 ]]; then
        PICKED_GPU="${gpu}"
        return 0
      fi
    else
      GPU_STREAK[${gpu}]=0
    fi
  done
  return 1
}

write_state running
log "queue started components=${COMPONENTS} gpu_ids=${GPU_IDS}"

while true; do
  refresh_jobs
  all_done=true
  any_failed=false
  for component in "${COMPONENT_LIST[@]}"; do
    state="${OUTPUT_ROOT}/${component}/state.txt"
    if [[ -f "${state}" ]] && grep -q '^status=failed$' "${state}"; then
      any_failed=true
    elif [[ ! -f "${state}" ]] || ! grep -q '^status=completed$' "${state}"; then
      all_done=false
    fi
  done
  if [[ "${all_done}" == true ]]; then
    if [[ "${any_failed}" == true ]]; then
      write_state completed_with_failures
      log "diagnostics finished with one or more failed components"
      exit 1
    fi
    write_state completed
    log "all diagnostics completed"
    exit 0
  fi

  while [[ "$(running_count)" -lt "${MAX_PARALLEL}" ]]; do
    next_component=""
    for component in "${COMPONENT_LIST[@]}"; do
      state="${OUTPUT_ROOT}/${component}/state.txt"
      if [[ -n "${JOB_PID[${component}]:-}" ]]; then
        continue
      fi
      if [[ -f "${state}" ]] && grep -Eq '^status=(completed|failed)$' "${state}"; then
        continue
      fi
      next_component="${component}"
      break
    done
    [[ -z "${next_component}" ]] && break
    pick_gpu || true
    gpu="${PICKED_GPU}"
    [[ -z "${gpu}" ]] && break

    component_log="${QUEUE_ROOT}/${next_component}.launcher.log"
    log "launching component=${next_component} gpu=${gpu}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export COMPONENT="${next_component}"
      export RUN_ID="${RUN_ID}"
      export OUTPUT_ROOT="${OUTPUT_ROOT}/${next_component}"
      export EXPERT_POLICY_PATH
      bash "${SCRIPT_DIR}/run_friend_flat_dr_component_diagnostic.sh"
    ) > "${component_log}" 2>&1 &
    JOB_PID[${next_component}]=$!
    JOB_GPU[${next_component}]="${gpu}"
    GPU_STREAK[${gpu}]=0
  done

  write_state running
  sleep "${POLL_SECONDS}"
done
