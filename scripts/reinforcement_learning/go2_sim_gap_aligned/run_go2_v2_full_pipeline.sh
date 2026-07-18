#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${DOMAIN:?Set DOMAIN to sim or real}"
: "${GAP_ID:?Set GAP_ID}"
: "${DATASET_PATH:?Set frozen V2 dataset path}"
: "${RUN_ROOT:?Set V2 output root}"
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU}"

case "${DOMAIN}" in sim|real) ;; *) echo "Invalid DOMAIN=${DOMAIN}" >&2; exit 2 ;; esac
case "${GAP_ID}" in g0|rr05|rr03|p5|p75) ;; *) echo "Invalid GAP_ID=${GAP_ID}" >&2; exit 2 ;; esac
case "${RUN_ROOT}" in *v2*contact18*matched25k*) ;; *) echo "RUN_ROOT lacks required V2 markers: ${RUN_ROOT}" >&2; exit 2 ;; esac

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EXPERT_POLICY="${EXPERT_POLICY:-${REPO_ROOT}/logs/experiments/go2_gap_experts/selected_g0d0/step97656}"
STATE_FILE="${RUN_ROOT}/state.txt"
RUN_LOG="${RUN_ROOT}/run.log"
RWM_ROOT="${RUN_ROOT}/rwm_baseline"
BASELINE_ROOT="${RUN_ROOT}/final_policy_rwm_p0"
TRACE_ROOT="${RUN_ROOT}/trace_t4_top25_dual"

mkdir -p "${RUN_ROOT}"
CURRENT_STAGE=preflight

write_state() {
  printf 'time=%s\ndomain=%s\ngap_id=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "${DOMAIN}" "${GAP_ID}" "$1" "$2" \
    "${CUDA_VISIBLE_DEVICES}" > "${STATE_FILE}"
}

run_logged() {
  local stage="$1"
  shift
  CURRENT_STAGE="${stage}"
  write_state "${stage}" running
  echo "[$(date --iso-8601=seconds)] start ${stage}" | tee -a "${RUN_LOG}"
  set +e
  "$@" 2>&1 | tee -a "${RUN_LOG}"
  local code=${PIPESTATUS[0]}
  set -e
  if (( code != 0 )); then
    write_state "${stage}" failed
    return "${code}"
  fi
  echo "[$(date --iso-8601=seconds)] complete ${stage}" | tee -a "${RUN_LOG}"
}

on_exit() {
  local code=$?
  if (( code != 0 )); then
    write_state "${CURRENT_STAGE}" failed
  fi
}
trap on_exit EXIT

for required in "${DATASET_PATH}" "${EXPERT_POLICY}/actor.pt" "${PYTHON_BIN}"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

if [[ ! -f "${RWM_ROOT}/stage/summary.json" ]]; then
  mkdir -p "${RWM_ROOT}/stage"
  run_logged train_rwm_v2 env \
    STAGE_DIR="${RWM_ROOT}/stage" OUTPUT_DIR="${RWM_ROOT}/runs" \
    DATASET_PATH="${DATASET_PATH}" DEVICE=cuda:0 SEED=200 \
    MAX_ITERATIONS=5000 BATCH_SIZE=1024 MICRO_BATCH_SIZE=256 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh
fi
MODEL_PATH="$(cat "${RWM_ROOT}/stage/artifact_path.txt")"

if [[ ! -f "${BASELINE_ROOT}/stage/summary.json" ]]; then
  mkdir -p "${BASELINE_ROOT}/stage"
  run_logged train_final_policy_rwm_v2 env \
    STAGE_DIR="${BASELINE_ROOT}/stage" OUTPUT_DIR="${BASELINE_ROOT}/run" \
    DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 \
    DEVICE=cuda:0 SEED=300 NUM_ENV_STEPS=50000000 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh
fi

run_logged train_trace_t4_top25_dual env \
  GAP_ID="${GAP_ID}" RUN_ROOT="${TRACE_ROOT}" TRACE_BATCH_ROOT="$(dirname "${RUN_ROOT}")" \
  OFFLINE_DATASET="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
  EXPERT_POLICY="${EXPERT_POLICY}" TRACE_ACTION_TEMPERATURE=4.0 \
  TRACE_SELECT_RATIO=0.25 \
  bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_sim_gap_trace_t4_dual_ratio_pipeline.sh

write_state done completed
cat > "${RUN_ROOT}/summary.txt" <<EOF
domain=${DOMAIN}
gap_id=${GAP_ID}
dataset=${DATASET_PATH}
rwm_model=${MODEL_PATH}
rwm_policy=$(cat "${BASELINE_ROOT}/stage/artifact_path.txt")
trace_r10=$(cat "${TRACE_ROOT}/final_policy_trace_r10/stage/artifact_path.txt")
trace_r25=$(cat "${TRACE_ROOT}/final_policy_trace_r25/stage/artifact_path.txt")
EOF
trap - EXIT
