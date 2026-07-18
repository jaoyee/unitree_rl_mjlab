#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${GAP_ID:?Set GAP_ID to g0, rr03, rr05, p5, or p75}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
for codex_dir in \
  /root/.vscode-server/extensions/openai.chatgpt-26.707.71524-linux-x64/bin/linux-x86_64 \
  /tmp/.X11-unix/codex_projects/runtime/codex-preflight
do
  if [[ -x "${codex_dir}/codex" ]]; then
    export PATH="${codex_dir}:${PATH}"
    break
  fi
done

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
BASE_ROOT="${BASE_ROOT:-${REPO_ROOT}/logs/experiments/go2_sim_gap_aligned/20260714_aligned_g0d0_v1}"
BASE_GAP_ROOT="${BASE_ROOT}/${GAP_ID}"
TRACE_ACTION_TEMPERATURE="${TRACE_ACTION_TEMPERATURE:-2.0}"
TRACE_SELECT_RATIO="${TRACE_SELECT_RATIO:-0.25}"
TRACE_ROLLOUT_LENGTH="${TRACE_ROLLOUT_LENGTH:-200}"
TRACE_TRAJECTORIES_PER_STATE="${TRACE_TRAJECTORIES_PER_STATE:-4}"
TRACE_START_STATE_COUNT="${TRACE_START_STATE_COUNT:-1024}"
TRACE_FAILURE_TRAJECTORY_RATIO="${TRACE_FAILURE_TRAJECTORY_RATIO:-0.20}"
TRACE_TERMINAL_PENALTY="${TRACE_TERMINAL_PENALTY:--10.0}"
TRACE_BATCH_ROOT="${TRACE_BATCH_ROOT:-$(dirname "${RUN_ROOT}")}"

case "${GAP_ID}" in
  g0)
    PAYLOAD_KG="0.0"
    RR_STRENGTH="1.0"
    ;;
  rr03)
    PAYLOAD_KG="0.0"
    RR_STRENGTH="0.3"
    ;;
  rr05)
    PAYLOAD_KG="0.0"
    RR_STRENGTH="0.5"
    ;;
  p5)
    PAYLOAD_KG="5.0"
    RR_STRENGTH="1.0"
    ;;
  p75)
    PAYLOAD_KG="7.5"
    RR_STRENGTH="1.0"
    ;;
  *)
    echo "Unsupported GAP_ID: ${GAP_ID}" >&2
    exit 2
    ;;
esac
OFFLINE_DATASET="${OFFLINE_DATASET:-${BASE_GAP_ROOT}/dataset_expert_command_25k.pt}"
EXPERT_POLICY="${EXPERT_POLICY:-${REPO_ROOT}/logs/experiments/go2_gap_experts/selected_g0d0/step97656}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$(cat "${BASE_GAP_ROOT}/rwm_baseline/stage/artifact_path.txt")"
fi

CANDIDATE_ENVS=$((TRACE_START_STATE_COUNT * TRACE_TRAJECTORIES_PER_STATE))
CANDIDATE_TRANSITIONS=$((CANDIDATE_ENVS * TRACE_ROLLOUT_LENGTH))
CANDIDATES="${RUN_ROOT}/imperfect_sim/candidates_t2_${TRACE_START_STATE_COUNT}x${TRACE_TRAJECTORIES_PER_STATE}x${TRACE_ROLLOUT_LENGTH}.pt"
SUMMARIES="${RUN_ROOT}/imperfect_sim/bootstrap.summaries.jsonl"
PAIRS="${RUN_ROOT}/feedback/pairs_500.jsonl"
LABEL_DIR="${RUN_ROOT}/feedback/labels"
LABELS="${LABEL_DIR}/codex_labels_filtered.jsonl"
SCORER="${RUN_ROOT}/feedback/go2_trace_scorer.pt"
REPLAY="${RUN_ROOT}/replay/selected_top25.pt"
STATE_FILE="${RUN_ROOT}/state.txt"
RUN_LOG="${RUN_ROOT}/run.log"
CODEX_LOCK="${TRACE_BATCH_ROOT}/codex_feedback.lock"

mkdir -p "${RUN_ROOT}" "$(dirname "${CANDIDATES}")" "$(dirname "${PAIRS}")" \
  "$(dirname "${REPLAY}")"

write_state() {
  local stage="$1" status="$2"
  printf 'time=%s\ngap_id=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "${GAP_ID}" "${stage}" "${status}" "${CUDA_VISIBLE_DEVICES}" \
    > "${STATE_FILE}"
}

run_stage() {
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

for required in "${OFFLINE_DATASET}" "${EXPERT_POLICY}/actor.pt" "${MODEL_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    write_state preflight failed
    exit 2
  fi
done

if [[ ! -f "${CANDIDATES}" ]]; then
  run_stage collect_imperfect_sim_candidates "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
    --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
    --device cuda:0 \
    --seed 42 \
    --num_envs "${CANDIDATE_ENVS}" \
    --num_transitions "${CANDIDATE_TRANSITIONS}" \
    --save_path "${CANDIDATES}" \
    --expert_policy_path "${EXPERT_POLICY}" \
    --collector_mix expert:0.80,random:0.20 \
    --fixed_collector_assignment \
    --collector_assignment_seed 42 \
    --dataset_obs_kind proprioceptive \
    --action_noise_std 0.0 \
    --failure_action_noise_std 0.0 \
    --randomization_preset calibrated_default \
    --randomization_components friction,mass_com,motor,delay,observation,push,initial_state \
    --randomization_scale 1.0 \
    --payload_mass_range_kg "${PAYLOAD_KG}" "${PAYLOAD_KG}" \
    --payload_position_body_m 0.0 0.0 0.10 \
    --payload_box_size_m 0.20 0.12 0.05 \
    --rr_calf_strength_range "${RR_STRENGTH}" "${RR_STRENGTH}" \
    --env_action_noise_std 0.0 \
    --env_action_bias_std 0.0 \
    --env_action_scale_range 1.0 1.0 \
    --env_action_delay_steps 0 0 \
    --chunk_size 0 \
    --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
    --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
    --x_range -0.5 0.5 \
    --signed_x \
    --x_abs_range 0.08 0.50 \
    --y_abs_range 0.05 0.20 \
    --yaw_abs_range 0.08 0.40 \
    --command_resample_interval_min 120 \
    --command_resample_interval_max 300 \
    --use_domain_randomization \
    --use_push_randomization \
    --use_observation_noise \
    --trace_reset_dataset "${OFFLINE_DATASET}" \
    --trace_rollout_length "${TRACE_ROLLOUT_LENGTH}" \
    --trace_trajectories_per_state "${TRACE_TRAJECTORIES_PER_STATE}" \
    --trace_action_temperature "${TRACE_ACTION_TEMPERATURE}" \
    --headless
fi

if [[ ! -f "${SUMMARIES}" ]]; then
  run_stage build_candidate_summaries "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/build_trace_replay_v3.py \
    --candidate_dataset "${CANDIDATES}" \
    --summaries_only \
    --trajectory_length "${TRACE_ROLLOUT_LENGTH}" \
    --trajectory_stride "${TRACE_ROLLOUT_LENGTH}" \
    --include_terminal_prefixes \
    --minimum_terminal_length 20 \
    --terminal_penalty "${TRACE_TERMINAL_PENALTY}" \
    --reward_source rwm_aligned \
    --output "${RUN_ROOT}/imperfect_sim/bootstrap.pt"
fi

if [[ ! -f "${PAIRS}" ]]; then
  run_stage build_feedback_pairs "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/build_go2_feedback_pairs.py \
    --summaries "${SUMMARIES}" \
    --output "${PAIRS}" \
    --budget 500 \
    --cross_start_fraction 0.20 \
    --seed 42
fi

filtered_label_count() {
  [[ -f "${LABELS}" ]] || { echo 0; return; }
  grep -cve '^[[:space:]]*$' "${LABELS}" || true
}

while (( $(filtered_label_count) < 200 )); do
  write_state waiting_for_codex_auth waiting
  echo "[$(date --iso-8601=seconds)] labeling attempt; current filtered labels=$(filtered_label_count)" \
    | tee -a "${RUN_LOG}"
  "${PYTHON_BIN}" scripts/reinforcement_learning/rwm_trace/label_feedback_with_codex.py \
    --prompts "${PAIRS}" \
    --output-dir "${LABEL_DIR}" \
    --schema scripts/reinforcement_learning/rwm_trace/feedback_label_batch.schema.json \
    --repo-root "${REPO_ROOT}" \
    --batch-size 20 \
    --confidence-threshold 0.70 \
    --max-retries 2 \
    --timeout-sec 300 \
    --lock-path "${CODEX_LOCK}" \
    --codex-service-tier default \
    2>&1 | tee -a "${RUN_LOG}" || true
  if (( $(filtered_label_count) < 200 )); then
    echo "[$(date --iso-8601=seconds)] insufficient labels; retrying in 600 seconds" | tee -a "${RUN_LOG}"
    sleep 600
  fi
done

if [[ ! -f "${SCORER}" ]]; then
  run_stage train_bt_scorer "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
    --labels "${LABELS}" \
    --pairs "${PAIRS}" \
    --output "${SCORER}" \
    --epochs 50 \
    --learning_rate 0.001 \
    --hidden_dim 256 \
    --confidence_threshold 0.70 \
    --val_ratio 0.20 \
    --seed 42
fi

SCORER_GATE="$(${PYTHON_BIN} - "${SCORER}" <<'PY'
import math
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
metadata = payload.get("metadata") or {}
num_val = int(metadata.get("num_val_pairs", 0))
val_accuracy = float(metadata.get("val_accuracy", float("nan")))
print(f"num_val_pairs={num_val} val_accuracy={val_accuracy:.6f}")
if num_val < 40 or not math.isfinite(val_accuracy) or val_accuracy < 0.60:
    raise SystemExit(3)
PY
)" || {
  write_state scorer_quality_gate failed
  echo "TRACE scorer failed quality gate: ${SCORER_GATE}" | tee -a "${RUN_LOG}"
  exit 3
}
echo "TRACE scorer quality gate passed: ${SCORER_GATE}" | tee -a "${RUN_LOG}"

if [[ ! -f "${REPLAY}" ]]; then
  run_stage build_score_selected_replay "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/build_trace_replay_v3.py \
    --candidate_dataset "${CANDIDATES}" \
    --scorer_checkpoint "${SCORER}" \
    --selection scorer \
    --select_ratio "${TRACE_SELECT_RATIO}" \
    --trajectory_length "${TRACE_ROLLOUT_LENGTH}" \
    --trajectory_stride "${TRACE_ROLLOUT_LENGTH}" \
    --include_terminal_prefixes \
    --minimum_terminal_length 20 \
    --failure_trajectory_ratio "${TRACE_FAILURE_TRAJECTORY_RATIO}" \
    --terminal_penalty "${TRACE_TERMINAL_PENALTY}" \
    --n_step 3 \
    --gamma 0.99 \
    --reward_source rwm_aligned \
    --seed 42 \
    --output "${REPLAY}"
fi

REPLAY_GATE="$("${PYTHON_BIN}" - "${REPLAY}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
metadata = payload.get("metadata") or {}
selected = int(metadata.get("selected_trajectory_count", 0))
terminal = int(metadata.get("selected_terminal_trajectory_count", 0))
terminal_rows = int(payload["terminated"].bool().sum())
reward_min = float(payload["reward"].min())
ratio = terminal / max(selected, 1)
print(
    f"selected={selected} terminal_trajectories={terminal} "
    f"terminal_ratio={ratio:.6f} terminal_rows={terminal_rows} reward_min={reward_min:.6f}"
)
if selected < 1 or terminal < 32 or ratio < 0.05 or terminal_rows < terminal or reward_min > -5.0:
    raise SystemExit(3)
PY
)" || {
  write_state replay_failure_quality_gate failed
  echo "TRACE replay failed failure-information gate: ${REPLAY_GATE}" | tee -a "${RUN_LOG}"
  exit 3
}
echo "TRACE replay failure-information gate passed: ${REPLAY_GATE}" | tee -a "${RUN_LOG}"

train_policy_ratio() {
  local ratio_id="$1" ratio="$2"
  local policy_root="${RUN_ROOT}/final_policy_trace_${ratio_id}"
  local policy_stage="${policy_root}/stage"
  local policy_output="${policy_root}/run"
  local validation_root="${policy_root}/mjlab_validation"
  mkdir -p "${policy_stage}"
  if [[ -f "${policy_stage}/summary.json" ]]; then
    echo "[reuse] completed final policy training ${ratio_id}" | tee -a "${RUN_LOG}"
  else
    write_state "train_final_policy_trace_${ratio_id}" running
    set +e
    STAGE_DIR="${policy_stage}" OUTPUT_DIR="${policy_output}" \
      DATASET_PATH="${OFFLINE_DATASET}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 \
      DEVICE=cuda:0 SEED=300 NUM_ENV_STEPS=50000000 \
      LIN_VEL_X_RANGE="-0.5 0.5" LIN_VEL_Y_RANGE="-0.2 0.2" \
      ANG_VEL_Z_RANGE="-0.4 0.4" \
      TRACE_REPLAY_PATH="${REPLAY}" TRACE_REPLAY_RATIO="${ratio}" \
      bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
      2>&1 | tee -a "${RUN_LOG}"
    local code=${PIPESTATUS[0]}
    set -e
    if (( code != 0 )); then
      write_state "train_final_policy_trace_${ratio_id}" failed
      return "${code}"
    fi
  fi

  run_stage "validate_and_select_${ratio_id}" env \
    POLICY_OUTPUT_DIR="${policy_output}" GAP_ID="${GAP_ID}" \
    VALIDATION_ROOT="${validation_root}" VALIDATION_STRIDE=5000 \
    VALIDATION_NUM_ENVS=128 VALIDATION_STEPS=2400 VALIDATION_SEED=401 \
    bash scripts/reinforcement_learning/go2_sim_gap_aligned/select_go2_policy_checkpoint_mjlab.sh
  local best_checkpoint
  best_checkpoint="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' \
    "${validation_root}/selection.json")"
  printf '%s\n' "${best_checkpoint}" > "${policy_stage}/artifact_path.txt"
  sha256sum "${best_checkpoint}/actor.pt" | awk '{print $1}' > "${policy_stage}/artifact_sha256.txt"
  printf '%s\n' "${best_checkpoint}" > "${policy_stage}/mjlab_selected_artifact_path.txt"
}

train_policy_ratio r10 0.10
train_policy_ratio r25 0.25

write_state done completed
cat > "${RUN_ROOT}/summary.txt" <<EOF
gap_id=${GAP_ID}
offline_dataset=${OFFLINE_DATASET}
frozen_rwm=${MODEL_PATH}
candidate_dataset=${CANDIDATES}
labels=${LABELS}
scorer=${SCORER}
selected_replay=${REPLAY}
trace_action_temperature=${TRACE_ACTION_TEMPERATURE}
trace_rollout_length=${TRACE_ROLLOUT_LENGTH}
trace_start_state_count=${TRACE_START_STATE_COUNT}
trace_trajectories_per_state=${TRACE_TRAJECTORIES_PER_STATE}
trace_failure_trajectory_ratio=${TRACE_FAILURE_TRAJECTORY_RATIO}
trace_terminal_penalty=${TRACE_TERMINAL_PENALTY}
trace_select_ratio=${TRACE_SELECT_RATIO}
final_policy_r10=$(cat "${RUN_ROOT}/final_policy_trace_r10/stage/artifact_path.txt")
final_policy_r25=$(cat "${RUN_ROOT}/final_policy_trace_r25/stage/artifact_path.txt")
EOF
