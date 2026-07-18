#!/usr/bin/env bash
set -u

REPO=/root/unitree_rl_mjlab_normal_fixstand_v2
V2_ROOT="${REPO}/logs/experiments/go2_v2_contact18_matched25k/real/20260715_v2_contact18_matched25k_formal"
V3_PILOT="${REPO}/logs/experiments/go2_v3_t2_long_failure/real/20260716_t2_long200_failure_pilot"
V3_ROOT="${REPO}/logs/experiments/go2_v3_t2_long_failure/real/20260716_t2_long200_failure_formal"
TRANSFER="${REPO}/logs/transfer_real_trace_inputs"
TRACE_NAME=trace_t2_long200_failure20_top25_dual
PYTHON_BIN="${REPO}/.venv/bin/python"
EXPERT_POLICY="${REPO}/logs/experiments/go2_gap_experts/selected_g0d0/step97656"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

cd "${REPO}" || exit 2
mkdir -p "${V3_ROOT}"
QUEUE_ID="${QUEUE_ID:-gpu${CUDA_VISIBLE_DEVICES}}"
QUEUE_LOG="${V3_ROOT}/queue_${QUEUE_ID}.log"
QUEUE_STATE="${V3_ROOT}/queue_${QUEUE_ID}_state.txt"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${QUEUE_LOG}"
}

state() {
  printf 'time=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "$1" "$2" "${CUDA_VISIBLE_DEVICES}" > "${QUEUE_STATE}"
}

model_for() {
  case "$1" in
    g0|rr05)
      echo "${TRANSFER}/$1/model_5000.pt"
      ;;
    rr03|p5|p75)
      cat "${V2_ROOT}/$1/rwm_baseline/stage/artifact_path.txt"
      ;;
    *)
      return 2
      ;;
  esac
}

train_ratio() {
  local gap="$1"
  local trace_root="$2"
  local ratio_id="$3"
  local ratio="$4"
  local dataset="${TRANSFER}/${gap}/dataset.pt"
  local model
  model="$(model_for "${gap}")" || return 2
  local replay="${trace_root}/replay/selected_top25.pt"
  local policy_root="${trace_root}/final_policy_trace_${ratio_id}"
  local stage_dir="${policy_root}/stage"
  local output_dir="${policy_root}/run"
  local validation_root="${policy_root}/mjlab_validation"

  if [[ ! -f "${replay}" ]]; then
    log "skip ${gap}-${ratio_id}: replay missing"
    return 3
  fi

  mkdir -p "${stage_dir}"
  if [[ ! -f "${stage_dir}/summary.json" ]]; then
    state "${gap}_${ratio_id}_train" running
    log "start ${gap}-${ratio_id} policy training"
    STAGE_DIR="${stage_dir}" OUTPUT_DIR="${output_dir}" \
      DATASET_PATH="${dataset}" MODEL_PATH="${model}" P_ID=P0 \
      DEVICE=cuda:0 SEED=300 NUM_ENV_STEPS=50000000 \
      LIN_VEL_X_RANGE="-0.5 0.5" LIN_VEL_Y_RANGE="-0.2 0.2" \
      ANG_VEL_Z_RANGE="-0.4 0.4" \
      TRACE_REPLAY_PATH="${replay}" TRACE_REPLAY_RATIO="${ratio}" \
      bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
      2>&1 | tee -a "${trace_root}/run.log"
    local train_code=${PIPESTATUS[0]}
    if (( train_code != 0 )); then
      log "failed ${gap}-${ratio_id} training code=${train_code}"
      return "${train_code}"
    fi
  else
    log "reuse completed ${gap}-${ratio_id} training"
  fi

  state "${gap}_${ratio_id}_validate" running
  log "validate all checkpoints for ${gap}-${ratio_id}"
  POLICY_OUTPUT_DIR="${output_dir}" GAP_ID="${gap}" \
    VALIDATION_ROOT="${validation_root}" VALIDATION_STRIDE=5000 \
    VALIDATION_NUM_ENVS=128 VALIDATION_STEPS=2400 VALIDATION_SEED=401 \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/select_go2_policy_checkpoint_mjlab.sh \
    2>&1 | tee -a "${trace_root}/run.log"
  local validation_code=${PIPESTATUS[0]}
  if (( validation_code != 0 )); then
    log "${gap}-${ratio_id} strict validation gate did not pass; metrics retained"
  else
    "${PYTHON_BIN}" - "${validation_root}/selection.json" "${stage_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
stage = Path(sys.argv[2])
checkpoint = Path(selection["best_checkpoint"])
(stage / "artifact_path.txt").write_text(str(checkpoint) + "\n")
digest = hashlib.sha256((checkpoint / "actor.pt").read_bytes()).hexdigest()
(stage / "artifact_sha256.txt").write_text(digest + "\n")
(stage / "mjlab_selected_artifact_path.txt").write_text(str(checkpoint) + "\n")
PY
  fi
  return 0
}

build_trace_and_train_r10() {
  local gap="$1"
  local run_root="${V3_ROOT}/${gap}/${TRACE_NAME}"
  local dataset="${TRANSFER}/${gap}/dataset.pt"
  local model
  model="$(model_for "${gap}")" || return 2
  mkdir -p "${run_root}"

  state "${gap}_trace_assets_and_r10" running
  log "start ${gap} T2/rollout200 candidate, feedback, replay and r10 pipeline"
  GAP_ID="${gap}" RUN_ROOT="${run_root}" TRACE_BATCH_ROOT="${V3_ROOT}" \
    OFFLINE_DATASET="${dataset}" MODEL_PATH="${model}" \
    EXPERT_POLICY="${EXPERT_POLICY}" TRACE_ACTION_TEMPERATURE=2.0 \
    TRACE_ROLLOUT_LENGTH=200 TRACE_START_STATE_COUNT=1024 \
    TRACE_TRAJECTORIES_PER_STATE=4 TRACE_SELECT_RATIO=0.25 \
    TRACE_FAILURE_TRAJECTORY_RATIO=0.20 TRACE_TERMINAL_PENALTY=-10.0 \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v3_t2_long_failure_pipeline.sh \
    2>&1 | tee -a "${run_root}/queue_wrapper.log"
  local pipeline_code=${PIPESTATUS[0]}
  log "${gap} inner pipeline returned code=${pipeline_code}; continuing queue"

  # The inner pipeline stops when a strict checkpoint gate fails. Reuse its
  # completed r10 artifacts and make sure all r10 metrics are retained.
  train_ratio "${gap}" "${run_root}" r10 0.10 || true
  train_ratio "${gap}" "${run_root}" r25 0.25 || true
}

if [[ "${RUN_G0_R25:-1}" == "1" ]]; then
  state g0_r25 running
  train_ratio g0 "${V3_PILOT}/g0/${TRACE_NAME}" r25 0.25 || true
fi

for gap in ${GAPS:-rr05 rr03 p5 p75}; do
  build_trace_and_train_r10 "${gap}" || true
done

state done completed
log "all queued real V3 remaining experiments finished"
