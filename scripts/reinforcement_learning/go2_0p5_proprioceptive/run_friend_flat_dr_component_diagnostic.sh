#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${EXPERT_POLICY_PATH:?Set EXPERT_POLICY_PATH to the best 0.5-strength expert checkpoint.}"

COMPONENT="${COMPONENT:-nominal}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/diagnostics/go2_0p5_friend_dr_components/${RUN_ID}/${COMPONENT}}"
DATASET_PATH="${DATASET_PATH:-${OUTPUT_ROOT}/dataset.pt}"
WM_SAVE_BASE="${WM_SAVE_BASE:-${OUTPUT_ROOT}/world_model}"
STATE_FILE="${OUTPUT_ROOT}/state.txt"
SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-256}"
COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS:-200000}"
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-100000}"
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-1000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-512}"
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda:0}"

USE_DOMAIN_RANDOMIZATION=false
USE_PUSH_RANDOMIZATION=false
USE_OBSERVATION_NOISE=false
RANDOMIZATION_PRESET=default
RANDOMIZATION_COMPONENTS=""
RANDOMIZATION_SCALE=1.0

case "${COMPONENT}" in
  nominal)
    ;;
  friction|mass_com|motor|delay)
    USE_DOMAIN_RANDOMIZATION=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS="${COMPONENT}"
    ;;
  push)
    USE_PUSH_RANDOMIZATION=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS=push
    ;;
  observation)
    USE_OBSERVATION_NOISE=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS=observation
    ;;
  full)
    USE_DOMAIN_RANDOMIZATION=true
    USE_PUSH_RANDOMIZATION=true
    USE_OBSERVATION_NOISE=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS=all
    ;;
  core_phys)
    USE_DOMAIN_RANDOMIZATION=true
    USE_PUSH_RANDOMIZATION=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS=friction,mass_com,motor,push
    ;;
  core_phys_obs)
    USE_DOMAIN_RANDOMIZATION=true
    USE_PUSH_RANDOMIZATION=true
    USE_OBSERVATION_NOISE=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS=friction,mass_com,motor,push,observation
    ;;
  core_phys_half)
    USE_DOMAIN_RANDOMIZATION=true
    USE_PUSH_RANDOMIZATION=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS=friction,mass_com,motor,push
    RANDOMIZATION_SCALE=0.5
    ;;
  core_phys_half_obs)
    USE_DOMAIN_RANDOMIZATION=true
    USE_PUSH_RANDOMIZATION=true
    USE_OBSERVATION_NOISE=true
    RANDOMIZATION_PRESET=friend_flat
    RANDOMIZATION_COMPONENTS=friction,mass_com,motor,push,observation
    RANDOMIZATION_SCALE=0.5
    ;;
  *)
    echo "Unknown COMPONENT=${COMPONENT}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}"

write_state() {
  local stage="$1"
  local status="$2"
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "run_id=${RUN_ID}"
    echo "component=${COMPONENT}"
    echo "stage=${stage}"
    echo "status=${status}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
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
  echo "expert_policy_path=${EXPERT_POLICY_PATH}"
  echo "component=${COMPONENT}"
  echo "dataset_path=${DATASET_PATH}"
  echo "wm_save_base=${WM_SAVE_BASE}"
  echo "collect_num_envs=${COLLECT_NUM_ENVS}"
  echo "collect_num_transitions=${COLLECT_NUM_TRANSITIONS}"
  echo "wm_max_iterations=${WM_MAX_ITERATIONS}"
  echo "use_domain_randomization=${USE_DOMAIN_RANDOMIZATION}"
  echo "use_push_randomization=${USE_PUSH_RANDOMIZATION}"
  echo "use_observation_noise=${USE_OBSERVATION_NOISE}"
  echo "randomization_preset=${RANDOMIZATION_PRESET}"
  echo "randomization_components=${RANDOMIZATION_COMPONENTS}"
  echo "randomization_scale=${RANDOMIZATION_SCALE}"
  echo "started_at=$(date --iso-8601=seconds)"
} > "${SUMMARY_FILE}"

CURRENT_STAGE=collect_dataset
if [[ ! -f "${DATASET_PATH}" ]]; then
  write_state "${CURRENT_STAGE}" running
  EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  COLLECT_DEVICE="${DEVICE}" \
  COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS}" \
  COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS}" \
  COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE}" \
  USE_DOMAIN_RANDOMIZATION="${USE_DOMAIN_RANDOMIZATION}" \
  USE_PUSH_RANDOMIZATION="${USE_PUSH_RANDOMIZATION}" \
  USE_OBSERVATION_NOISE="${USE_OBSERVATION_NOISE}" \
  RANDOMIZATION_PRESET="${RANDOMIZATION_PRESET}" \
  RANDOMIZATION_COMPONENTS="${RANDOMIZATION_COMPONENTS}" \
  RANDOMIZATION_SCALE="${RANDOMIZATION_SCALE}" \
  RESET_PARTS=true \
  bash "${SCRIPT_DIR}/02_collect_expert_dataset.sh" \
    2>&1 | tee "${OUTPUT_ROOT}/01_collect_dataset.log"
fi

CURRENT_STAGE=train_rwm
FINAL_METRICS="$(find "${WM_SAVE_BASE}" -mindepth 2 -maxdepth 2 -name final_metrics.json -print -quit 2>/dev/null || true)"
if [[ -z "${FINAL_METRICS}" ]]; then
  write_state "${CURRENT_STAGE}" running
  DATASET_PATH="${DATASET_PATH}" \
  WM_SAVE_BASE="${WM_SAVE_BASE}" \
  WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS}" \
  WM_BATCH_SIZE="${WM_BATCH_SIZE}" \
  WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE}" \
  WM_DEVICE="${DEVICE}" \
  WM_SAVE_INTERVAL="${WM_MAX_ITERATIONS}" \
  WM_LOG_INTERVAL=50 \
  bash "${SCRIPT_DIR}/03_train_rwm_unmasked.sh" \
    2>&1 | tee "${OUTPUT_ROOT}/02_train_rwm.log"
fi

write_state completed completed
echo "completed_at=$(date --iso-8601=seconds)" >> "${SUMMARY_FILE}"
