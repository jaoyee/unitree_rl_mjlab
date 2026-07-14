#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${DATASET_PATH:?Set DATASET_PATH to the successful no-DR 0.5 dataset.}"

RUN_ID="${RUN_ID:-nodr_regression_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/overnight/${RUN_ID}}"
WM_SAVE_BASE="${WM_SAVE_BASE:-${OUTPUT_ROOT}/world_model}"
SAC_SAVE_PATH="${SAC_SAVE_PATH:-${OUTPUT_ROOT}/final_policy/TIMESTAMP}"
STATE_FILE="${OUTPUT_ROOT}/state.txt"
SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-5000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE:-256}"
SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS:-1024}"
SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS:-50000000}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUTPUT_ROOT}"

write_state() {
  local stage="$1"
  local status="$2"
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "run_id=${RUN_ID}"
    echo "stage=${stage}"
    echo "status=${status}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "dataset_path=${DATASET_PATH}"
  } > "${STATE_FILE}"
}

on_error() {
  local exit_code=$?
  write_state "${CURRENT_STAGE:-unknown}" failed
  echo "exit_code=${exit_code}" >> "${STATE_FILE}"
  exit "${exit_code}"
}
trap on_error ERR

{
  echo "repo_root=${REPO_ROOT}"
  echo "dataset_path=${DATASET_PATH}"
  echo "wm_save_base=${WM_SAVE_BASE}"
  echo "sac_save_path=${SAC_SAVE_PATH}"
  echo "wm_max_iterations=${WM_MAX_ITERATIONS}"
  echo "sac_num_env_steps=${SAC_NUM_ENV_STEPS}"
  echo "started_at=$(date --iso-8601=seconds)"
} > "${SUMMARY_FILE}"

CURRENT_STAGE=train_rwm
WM_FINAL_METRICS="$(find "${WM_SAVE_BASE}" -mindepth 2 -maxdepth 2 -name final_metrics.json -print -quit 2>/dev/null || true)"
if [[ -z "${WM_FINAL_METRICS}" ]]; then
  write_state "${CURRENT_STAGE}" running
  DATASET_PATH="${DATASET_PATH}" \
  WM_SAVE_BASE="${WM_SAVE_BASE}" \
  WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS}" \
  WM_BATCH_SIZE="${WM_BATCH_SIZE}" \
  WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE}" \
  WM_DEVICE="${DEVICE}" \
  bash "${SCRIPT_DIR}/03_train_rwm_unmasked.sh" \
    2>&1 | tee "${OUTPUT_ROOT}/01_train_rwm.log"
fi

CURRENT_STAGE=train_final_policy
if ! find "${OUTPUT_ROOT}/final_policy" -type f -path '*/step*/actor.pt' -print -quit 2>/dev/null | grep -q .; then
  write_state "${CURRENT_STAGE}" running
  DATASET_PATH="${DATASET_PATH}" \
  WM_SAVE_BASE="${WM_SAVE_BASE}" \
  WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS}" \
  SAC_SAVE_PATH="${SAC_SAVE_PATH}" \
  SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS}" \
  SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS}" \
  SAC_DEVICE="${DEVICE}" \
  bash "${SCRIPT_DIR}/05_train_policy_unmasked.sh" \
    2>&1 | tee "${OUTPUT_ROOT}/02_train_final_policy.log"
fi

write_state completed completed
echo "completed_at=$(date --iso-8601=seconds)" >> "${SUMMARY_FILE}"
