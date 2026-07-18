#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/unitree_rl_mjlab_normal_fixstand_v2
RUN_ROOT="${REPO_ROOT}/logs/experiments/go2_real_trace_rwm/20260714_g0_25k_v1"
OFFLINE_DATASET="${REPO_ROOT}/logs/rwm_datasets/go2_real_g0_baseline_25k_20260714/dataset.pt"
EXPERT_POLICY="${REPO_ROOT}/logs/experiments/go2_gap_experts/selected_g0d0/step97656"
RWM_STAGE="${RUN_ROOT}/rwm_baseline/stage"
RWM_EXIT_FILE="${RUN_ROOT}/rwm_exit_code.txt"

CANDIDATES="${RUN_ROOT}/imperfect_sim/candidates_1024x4x20.pt"
SUMMARIES="${RUN_ROOT}/imperfect_sim/bootstrap.summaries.jsonl"
PAIRS="${RUN_ROOT}/feedback/pairs_500.jsonl"
LABEL_DIR="${RUN_ROOT}/feedback/labels"
LABELS="${LABEL_DIR}/codex_labels_filtered.jsonl"
SCORER="${RUN_ROOT}/feedback/go2_trace_scorer.pt"
REPLAY="${RUN_ROOT}/replay/selected_top10.pt"
POLICY_STAGE="${RUN_ROOT}/final_policy_trace/stage"
POLICY_OUTPUT="${RUN_ROOT}/final_policy_trace/run"
STATE_FILE="${RUN_ROOT}/trace_state.txt"
RUN_LOG="${RUN_ROOT}/trace.log"
CODEX_LOCK="${REPO_ROOT}/logs/experiments/go2_sim_gap_aligned/20260714_aligned_g0d0_v1/trace_method_v1/codex_feedback.lock"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PATH="/root/.vscode-server/extensions/openai.chatgpt-26.707.71524-linux-x64/bin/linux-x86_64:${PATH}"

mkdir -p "${RUN_ROOT}" "$(dirname "${CANDIDATES}")" "$(dirname "${PAIRS}")" \
  "$(dirname "${REPLAY}")" "${POLICY_STAGE}"
cd "${REPO_ROOT}"

write_state() {
  printf 'time=%s\nstage=%s\nstatus=%s\ngpu=1\n' \
    "$(date --iso-8601=seconds)" "$1" "$2" > "${STATE_FILE}"
}

run_stage() {
  local stage="$1"
  shift
  write_state "${stage}" running
  echo "[$(date --iso-8601=seconds)] start ${stage}" | tee -a "${RUN_LOG}"
  "$@" 2>&1 | tee -a "${RUN_LOG}"
  local code=${PIPESTATUS[0]}
  if (( code != 0 )); then
    write_state "${stage}" failed
    return "${code}"
  fi
  echo "[$(date --iso-8601=seconds)] complete ${stage}" | tee -a "${RUN_LOG}"
}

for required in "${OFFLINE_DATASET}" "${EXPERT_POLICY}/actor.pt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    write_state preflight failed
    exit 2
  fi
done

if [[ ! -f "${CANDIDATES}" ]]; then
  run_stage collect_real_reset_imperfect_sim_candidates "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
    --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
    --device cuda:0 --seed 42 --num_envs 4096 --num_transitions 81920 \
    --save_path "${CANDIDATES}" --expert_policy_path "${EXPERT_POLICY}" \
    --collector_mix expert:1.0 --dataset_obs_kind proprioceptive --action_noise_std 0.0 \
    --randomization_preset calibrated_default \
    --randomization_components friction,mass_com,motor,delay,observation,push,initial_state \
    --randomization_scale 1.0 \
    --payload_mass_range_kg 0.0 0.0 \
    --payload_position_body_m 0.0 0.0 0.10 \
    --payload_box_size_m 0.20 0.12 0.05 \
    --rr_calf_strength_range 1.0 1.0 \
    --env_action_noise_std 0.0 --env_action_bias_std 0.0 \
    --env_action_scale_range 1.0 1.0 --env_action_delay_steps 0 0 --chunk_size 0 \
    --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
    --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
    --x_range -0.8 0.8 --signed_x --x_abs_range 0.08 0.80 \
    --y_abs_range 0.05 0.30 --yaw_abs_range 0.08 0.60 \
    --command_resample_interval_min 120 --command_resample_interval_max 300 \
    --use_domain_randomization --use_push_randomization --use_observation_noise \
    --trace_reset_dataset "${OFFLINE_DATASET}" \
    --trace_rollout_length 20 --trace_trajectories_per_state 4 --headless
fi

if [[ ! -f "${SUMMARIES}" ]]; then
  run_stage build_candidate_summaries "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/build_trace_replay.py \
    --candidate_dataset "${CANDIDATES}" --summaries_only \
    --trajectory_length 20 --trajectory_stride 20 --reward_source rwm_aligned \
    --output "${RUN_ROOT}/imperfect_sim/bootstrap.pt"
fi

if [[ ! -f "${PAIRS}" ]]; then
  run_stage build_feedback_pairs "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/build_go2_feedback_pairs.py \
    --summaries "${SUMMARIES}" --output "${PAIRS}" --budget 500 \
    --cross_start_fraction 0.20 --seed 42
fi

filtered_label_count() {
  [[ -f "${LABELS}" ]] || { echo 0; return; }
  grep -cve '^[[:space:]]*$' "${LABELS}" || true
}

while (( $(filtered_label_count) < 200 )); do
  write_state labeling_codex waiting
  echo "[$(date --iso-8601=seconds)] labeling attempt; filtered=$(filtered_label_count)" | tee -a "${RUN_LOG}"
  "${PYTHON_BIN}" scripts/reinforcement_learning/rwm_trace/label_feedback_with_codex.py \
    --prompts "${PAIRS}" --output-dir "${LABEL_DIR}" \
    --schema scripts/reinforcement_learning/rwm_trace/feedback_label_batch.schema.json \
    --repo-root "${REPO_ROOT}" --batch-size 20 --confidence-threshold 0.70 \
    --max-retries 2 --timeout-sec 300 --lock-path "${CODEX_LOCK}" \
    --codex-service-tier default 2>&1 | tee -a "${RUN_LOG}" || true
  if (( $(filtered_label_count) < 200 )); then
    echo "[$(date --iso-8601=seconds)] insufficient labels; retrying in 600 seconds" | tee -a "${RUN_LOG}"
    sleep 600
  fi
done

if [[ ! -f "${SCORER}" ]]; then
  run_stage train_bt_scorer "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
    --labels "${LABELS}" --pairs "${PAIRS}" --output "${SCORER}" \
    --epochs 50 --learning_rate 0.001 --hidden_dim 256 \
    --confidence_threshold 0.70 --val_ratio 0.20 --seed 42
fi

SCORER_GATE="$(${PYTHON_BIN} - "${SCORER}" <<'PY'
import math
import sys
import torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = p.get("metadata") or {}
n = int(m.get("num_val_pairs", 0))
a = float(m.get("val_accuracy", float("nan")))
print(f"num_val_pairs={n} val_accuracy={a:.6f}")
if n < 40 or not math.isfinite(a) or a < 0.60:
    raise SystemExit(3)
PY
)" || { write_state scorer_quality_gate failed; exit 3; }
echo "TRACE scorer quality gate passed: ${SCORER_GATE}" | tee -a "${RUN_LOG}"

if [[ ! -f "${REPLAY}" ]]; then
  run_stage build_score_selected_replay "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_trace/build_trace_replay.py \
    --candidate_dataset "${CANDIDATES}" --scorer_checkpoint "${SCORER}" \
    --selection scorer --select_ratio 0.10 --trajectory_length 20 --trajectory_stride 20 \
    --n_step 3 --gamma 0.99 --reward_source rwm_aligned --seed 42 --output "${REPLAY}"
fi

write_state waiting_for_real_rwm waiting
while [[ ! -f "${RWM_EXIT_FILE}" ]]; do sleep 30; done
if [[ "$(tr -d '[:space:]' < "${RWM_EXIT_FILE}")" != 0 ]]; then
  echo "Real RWM stage failed; refusing to start policy." | tee -a "${RUN_LOG}"
  write_state waiting_for_real_rwm failed
  exit 4
fi
MODEL_PATH="$(cat "${RWM_STAGE}/artifact_path.txt")"
if [[ ! -s "${MODEL_PATH}" || ! -s "${RWM_STAGE}/summary.json" ]]; then
  echo "Real RWM artifacts incomplete." | tee -a "${RUN_LOG}"
  write_state validate_real_rwm failed
  exit 5
fi

if [[ ! -f "${POLICY_STAGE}/summary.json" ]]; then
  write_state train_final_policy_trace running
  STAGE_DIR="${POLICY_STAGE}" OUTPUT_DIR="${POLICY_OUTPUT}" \
    DATASET_PATH="${OFFLINE_DATASET}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 \
    DEVICE=cuda:0 SEED=300 NUM_ENV_STEPS=50000000 \
    TRACE_REPLAY_PATH="${REPLAY}" TRACE_REPLAY_RATIO=0.10 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
    2>&1 | tee -a "${RUN_LOG}"
  code=${PIPESTATUS[0]}
  if (( code != 0 )); then write_state train_final_policy_trace failed; exit "${code}"; fi
fi

write_state done completed
cat > "${RUN_ROOT}/summary.txt" <<EOF
source=real_go2_g0_baseline_25k
offline_dataset=${OFFLINE_DATASET}
frozen_rwm=${MODEL_PATH}
candidate_dataset=${CANDIDATES}
labels=${LABELS}
scorer=${SCORER}
selected_replay=${REPLAY}
final_policy=$(cat "${POLICY_STAGE}/artifact_path.txt")
EOF
