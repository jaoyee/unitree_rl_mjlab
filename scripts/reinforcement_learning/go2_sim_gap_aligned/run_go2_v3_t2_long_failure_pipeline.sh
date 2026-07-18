#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${DOMAIN:?Set DOMAIN to sim or real}"
: "${GAP_ID:?Set GAP_ID}"
: "${DATASET_PATH:?Set the frozen V2 dataset path}"
: "${V2_RUN_ROOT:?Set the completed V2 condition root whose frozen RWM is reused}"
: "${RUN_ROOT:?Set the independent V3 output root}"
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU}"

case "${DOMAIN}" in sim|real) ;; *) echo "Invalid DOMAIN=${DOMAIN}" >&2; exit 2 ;; esac
case "${GAP_ID}" in g0|rr03|rr05|p5|p75) ;; *) echo "Invalid GAP_ID=${GAP_ID}" >&2; exit 2 ;; esac
case "${RUN_ROOT}" in *v3*t2*long*failure*) ;; *) echo "RUN_ROOT lacks V3 markers: ${RUN_ROOT}" >&2; exit 2 ;; esac

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
BASELINE_ROOT="${RUN_ROOT}/final_policy_rwm_p0"
TRACE_ROOT="${RUN_ROOT}/trace_t2_long200_failure20_top25_dual"
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$(cat "${V2_RUN_ROOT}/rwm_baseline/stage/artifact_path.txt")"
fi

mkdir -p "${RUN_ROOT}"

write_state() {
  printf 'time=%s\ndomain=%s\ngap_id=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "${DOMAIN}" "${GAP_ID}" "$1" "$2" \
    "${CUDA_VISIBLE_DEVICES}" > "${STATE_FILE}"
}

run_logged() {
  local stage="$1"
  shift
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

for required in "${DATASET_PATH}" "${MODEL_PATH}" "${EXPERT_POLICY}/actor.pt" "${PYTHON_BIN}"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

if [[ ! -f "${BASELINE_ROOT}/stage/summary.json" ]]; then
  mkdir -p "${BASELINE_ROOT}/stage"
  run_logged train_command_aligned_p0 env \
    STAGE_DIR="${BASELINE_ROOT}/stage" OUTPUT_DIR="${BASELINE_ROOT}/run" \
    DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 \
    DEVICE=cuda:0 SEED=300 NUM_ENV_STEPS=50000000 \
    LIN_VEL_X_RANGE="-0.5 0.5" LIN_VEL_Y_RANGE="-0.2 0.2" \
    ANG_VEL_Z_RANGE="-0.4 0.4" \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh
fi

run_logged validate_and_select_p0 env \
  POLICY_OUTPUT_DIR="${BASELINE_ROOT}/run" GAP_ID="${GAP_ID}" \
  VALIDATION_ROOT="${BASELINE_ROOT}/mjlab_validation" VALIDATION_STRIDE=5000 \
  VALIDATION_NUM_ENVS=128 VALIDATION_STEPS=2400 VALIDATION_SEED=401 \
  bash scripts/reinforcement_learning/go2_sim_gap_aligned/select_go2_policy_checkpoint_mjlab.sh
BEST_P0="$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' \
  "${BASELINE_ROOT}/mjlab_validation/selection.json")"
printf '%s\n' "${BEST_P0}" > "${BASELINE_ROOT}/stage/artifact_path.txt"
sha256sum "${BEST_P0}/actor.pt" | awk '{print $1}' > "${BASELINE_ROOT}/stage/artifact_sha256.txt"

run_logged train_trace_t2_long_failure env \
  GAP_ID="${GAP_ID}" RUN_ROOT="${TRACE_ROOT}" TRACE_BATCH_ROOT="$(dirname "${RUN_ROOT}")" \
  OFFLINE_DATASET="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
  EXPERT_POLICY="${EXPERT_POLICY}" TRACE_ACTION_TEMPERATURE=2.0 \
  TRACE_ROLLOUT_LENGTH=200 TRACE_START_STATE_COUNT=1024 \
  TRACE_TRAJECTORIES_PER_STATE=4 TRACE_SELECT_RATIO=0.25 \
  TRACE_FAILURE_TRAJECTORY_RATIO=0.20 TRACE_TERMINAL_PENALTY=-10.0 \
  bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v3_t2_long_failure_pipeline.sh

write_state done completed
cat > "${RUN_ROOT}/summary.txt" <<EOF
domain=${DOMAIN}
gap_id=${GAP_ID}
dataset=${DATASET_PATH}
frozen_v2_rwm=${MODEL_PATH}
command_range=vx[-0.5,0.5],vy[-0.2,0.2],yaw[-0.4,0.4]
rwm_policy_mjlab_selected=$(cat "${BASELINE_ROOT}/stage/artifact_path.txt")
trace_r10_mjlab_selected=$(cat "${TRACE_ROOT}/final_policy_trace_r10/stage/artifact_path.txt")
trace_r25_mjlab_selected=$(cat "${TRACE_ROOT}/final_policy_trace_r25/stage/artifact_path.txt")
EOF
