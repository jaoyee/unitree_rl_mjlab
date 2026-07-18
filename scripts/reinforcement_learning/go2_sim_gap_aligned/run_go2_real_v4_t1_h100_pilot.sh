#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/unitree_rl_mjlab_normal_fixstand_v2}"
cd "${REPO}"

: "${GAP_ID:?Set GAP_ID to g0 or p5}"
case "${GAP_ID}" in
  g0)
    PAYLOAD_KG=0.0; RR_STRENGTH=1.0
    P0="${REPO}/logs/experiments/go2_v2_contact18_matched25k/real/20260715_v2_contact18_matched25k_formal/g0/final_policy_rwm_p0/run/step48828"
    DATASET="${REPO}/logs/transfer_real_trace_inputs/g0/dataset.pt"
    MODEL="${REPO}/logs/transfer_real_trace_inputs/g0/model_5000.pt"
    ;;
  p5)
    PAYLOAD_KG=5.0; RR_STRENGTH=1.0
    P0="${REPO}/logs/experiments/go2_v2_contact18_matched25k/real/20260715_v2_contact18_matched25k_formal/p5/final_policy_rwm_p0/run/step48828"
    DATASET="${REPO}/logs/transfer_real_trace_inputs/p5/dataset.pt"
    MODEL="${REPO}/logs/experiments/go2_v2_contact18_matched25k/real/20260715_v2_contact18_matched25k_formal/p5/rwm_baseline/runs/2026-07-16_04-03-21/model_5000.pt"
    ;;
  *) echo "Unsupported GAP_ID=${GAP_ID}" >&2; exit 2 ;;
esac

P0="${P0_OVERRIDE:-${P0}}"
DATASET="${DATASET_OVERRIDE:-${DATASET}}"
MODEL="${MODEL_OVERRIDE:-${MODEL}}"

MATRIX_ROOT="${REPO}/logs/experiments/go2_real_v4_t1_h100_r10/20260717_pilot"
RUN_ROOT="${MATRIX_ROOT}/${GAP_ID}"
PYTHON="${REPO}/.venv/bin/python"
TRAIN="${REPO}/scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh"

HORIZON=100
START_STATES=1024
TRAJECTORIES_PER_STATE=4
CANDIDATE_ENVS=$((START_STATES * TRAJECTORIES_PER_STATE))
CANDIDATE_TRANSITIONS=$((CANDIDATE_ENVS * HORIZON))
EXTRA_ENV_STEPS=10240000

CANDIDATES="${RUN_ROOT}/replay_source/candidates_${GAP_ID}_p0_t1_h100.pt"
SUMMARIES="${RUN_ROOT}/replay_source/bootstrap.summaries.jsonl"
PAIRS="${RUN_ROOT}/feedback/pairs_500.jsonl"
LABEL_DIR="${RUN_ROOT}/feedback/labels"
LABELS="${LABEL_DIR}/codex_labels_filtered.jsonl"
SCORER="${RUN_ROOT}/feedback/go2_trace_scorer.pt"
RANDOM_REPLAY="${RUN_ROOT}/replay/random25_failure20.pt"
TRACE_REPLAY="${RUN_ROOT}/replay/trace_top25_failure20.pt"
STATE="${RUN_ROOT}/state.txt"
LOG="${RUN_ROOT}/run.log"
CODEX_LOCK="${MATRIX_ROOT}/codex_feedback.lock"

export WANDB_MODE=offline
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
for codex_dir in \
  /root/.vscode-server/extensions/openai.chatgpt-26.707.71524-linux-x64/bin/linux-x86_64 \
  /tmp/.X11-unix/codex_projects/runtime/codex-preflight
do
  if [[ -x "${codex_dir}/codex" ]]; then
    export PATH="${codex_dir}:${PATH}"
    break
  fi
done

mkdir -p "${RUN_ROOT}" "$(dirname "${CANDIDATES}")" "${LABEL_DIR}" "$(dirname "${RANDOM_REPLAY}")"

write_state() {
  printf 'time=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "$1" "$2" "${CUDA_VISIBLE_DEVICES:-unset}" > "${STATE}"
}

run_stage() {
  local stage="$1"
  shift
  write_state "${stage}" running
  echo "[$(date --iso-8601=seconds)] start ${stage}" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
  echo "[$(date --iso-8601=seconds)] complete ${stage}" | tee -a "${LOG}"
}

for required in "${P0}/actor.pt" "${P0}/agent_state.pt" "${DATASET}" "${MODEL}" "${PYTHON}"; do
  [[ -e "${required}" ]] || { echo "Missing ${required}" >&2; exit 2; }
done

cat > "${RUN_ROOT}/protocol.txt" <<EOF
gap_id=${GAP_ID}
baseline_checkpoint=${P0}
dataset=${DATASET}
frozen_world_model=${MODEL}
temperature=1.0
rollout_horizon=100
candidate_policy=${GAP_ID}_P0
candidate_dr_components=friction,mass_com,motor,delay,initial_state
candidate_dr_scale=0.2
failure_trajectory_ratio=0.20
trace_selection_ratio=0.25
policy_replay_ratios=0.10
failure_backprop_steps=40
failure_backprop_penalty=2.0
action_saturation_threshold=0.95
max_saturation_fraction=0.30
action_saturation_penalty_scale=10.0
action_delta_penalty_scale=0.10
extra_env_steps_per_branch=${EXTRA_ENV_STEPS}
extra_interaction_steps_per_branch=$((EXTRA_ENV_STEPS / 1024))
seed=300
EOF
sha256sum "${P0}/actor.pt" "${P0}/critic.pt" "${P0}/agent_state.pt" "${DATASET}" "${MODEL}" \
  > "${RUN_ROOT}/input_sha256.txt"

if [[ ! -s "${CANDIDATES}" ]]; then
  run_stage "collect_${GAP_ID}_p0_candidates" "${PYTHON}" \
    scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
    --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
    --device cuda:0 --seed 42 --num_envs "${CANDIDATE_ENVS}" \
    --num_transitions "${CANDIDATE_TRANSITIONS}" --save_path "${CANDIDATES}" \
    --expert_policy_path "${P0}" --collector_mix expert:0.80,random:0.20 \
    --fixed_collector_assignment --collector_assignment_seed 42 \
    --dataset_obs_kind proprioceptive --action_noise_std 0.0 \
    --failure_action_noise_std 0.0 --randomization_preset calibrated_default \
    --randomization_components friction,mass_com,motor,delay,initial_state \
    --randomization_scale 0.2 --payload_mass_range_kg "${PAYLOAD_KG}" "${PAYLOAD_KG}" \
    --payload_position_body_m 0.0 0.0 0.10 --payload_box_size_m 0.20 0.12 0.05 \
    --rr_calf_strength_range "${RR_STRENGTH}" "${RR_STRENGTH}" --env_action_noise_std 0.0 \
    --env_action_bias_std 0.0 --env_action_scale_range 1.0 1.0 \
    --env_action_delay_steps 0 0 --chunk_size 0 \
    --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
    --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
    --x_range -0.5 0.5 --signed_x --x_abs_range 0.08 0.50 \
    --y_abs_range 0.05 0.20 --yaw_abs_range 0.08 0.40 \
    --command_resample_interval_min 100 --command_resample_interval_max 250 \
    --use_domain_randomization \
    --trace_reset_dataset "${DATASET}" --trace_rollout_length "${HORIZON}" \
    --trace_trajectories_per_state "${TRAJECTORIES_PER_STATE}" \
    --trace_action_temperature 1.0 --headless
fi

if [[ ! -s "${SUMMARIES}" ]]; then
  run_stage build_summaries "${PYTHON}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v4.py \
    --candidate_dataset "${CANDIDATES}" --summaries_only \
    --trajectory_length "${HORIZON}" --trajectory_stride "${HORIZON}" \
    --include_terminal_prefixes --minimum_terminal_length 20 \
    --terminal_penalty -10.0 --reward_source rwm_aligned \
    --output "${RUN_ROOT}/replay_source/bootstrap.pt"
fi

if [[ ! -s "${PAIRS}" ]]; then
  run_stage build_feedback_pairs "${PYTHON}" \
    scripts/reinforcement_learning/rwm_trace/build_go2_feedback_pairs.py \
    --summaries "${SUMMARIES}" --output "${PAIRS}" --budget 500 \
    --cross_start_fraction 0.20 --seed 42
fi

if [[ ! -s "${RANDOM_REPLAY}" ]]; then
  run_stage build_random_replay "${PYTHON}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v4.py \
    --candidate_dataset "${CANDIDATES}" --selection random --select_ratio 0.25 \
    --trajectory_length "${HORIZON}" --trajectory_stride "${HORIZON}" \
    --include_terminal_prefixes --minimum_terminal_length 20 \
    --failure_trajectory_ratio 0.20 --terminal_penalty -10.0 \
    --failure_backprop_steps 40 --failure_backprop_penalty 2.0 \
    --action_saturation_threshold 0.95 --max_saturation_fraction 0.30 \
    --action_saturation_penalty_scale 10.0 --action_delta_penalty_scale 0.10 \
    --n_step 3 --gamma 0.99 --reward_source rwm_aligned --seed 42 \
    --output "${RANDOM_REPLAY}"
fi

filtered_label_count() {
  [[ -f "${LABELS}" ]] || { echo 0; return; }
  grep -cve '^[[:space:]]*$' "${LABELS}" || true
}

label_until_ready() {
  while (( $(filtered_label_count) < 200 )); do
    echo "[$(date --iso-8601=seconds)] labeling current=$(filtered_label_count)" >> "${LOG}"
    "${PYTHON}" scripts/reinforcement_learning/rwm_trace/label_feedback_with_codex.py \
      --prompts "${PAIRS}" --output-dir "${LABEL_DIR}" \
      --schema scripts/reinforcement_learning/rwm_trace/feedback_label_batch.schema.json \
      --repo-root "${REPO}" --batch-size 20 --confidence-threshold 0.70 \
      --max-retries 2 --timeout-sec 300 --lock-path "${CODEX_LOCK}" \
      --codex-service-tier default >> "${LOG}" 2>&1 || true
    (( $(filtered_label_count) >= 200 )) || sleep 600
  done
}

if (( $(filtered_label_count) < 200 )); then
  label_until_ready &
  LABEL_PID=$!
else
  LABEL_PID=""
fi

train_branch() {
  local name="$1" replay="$2" ratio="$3"
  local root="${RUN_ROOT}/policies/${name}"
  if [[ -s "${root}/stage/summary.json" ]]; then
    echo "[reuse] ${name}" | tee -a "${LOG}"
    return
  fi
  args=(env POLICY_RESUME_PATH="${P0}" STAGE_DIR="${root}/stage" OUTPUT_DIR="${root}/run"
    DATASET_PATH="${DATASET}" MODEL_PATH="${MODEL}" P_ID=P0 DEVICE=cuda:0 SEED=300
    NUM_ENV_STEPS="${EXTRA_ENV_STEPS}" LIN_VEL_X_RANGE="-0.5 0.5"
    LIN_VEL_Y_RANGE="-0.2 0.2" ANG_VEL_Z_RANGE="-0.4 0.4")
  if [[ -n "${replay}" ]]; then
    args+=(TRACE_REPLAY_PATH="${replay}" TRACE_REPLAY_RATIO="${ratio}")
  fi
  run_stage "train_${name}" "${args[@]}" bash "${TRAIN}"
}

train_branch p0_continued "" 0
train_branch random_r10 "${RANDOM_REPLAY}" 0.10

if [[ -n "${LABEL_PID}" ]]; then
  write_state waiting_for_trace_labels waiting
  wait "${LABEL_PID}"
fi

if [[ ! -s "${SCORER}" ]]; then
  run_stage train_trace_scorer "${PYTHON}" \
    scripts/reinforcement_learning/rwm_trace/train_go2_trace_scorer.py \
    --labels "${LABELS}" --pairs "${PAIRS}" --output "${SCORER}" \
    --epochs 50 --learning_rate 0.001 --hidden_dim 256 \
    --confidence_threshold 0.70 --val_ratio 0.20 --seed 42
fi

"${PYTHON}" - "${SCORER}" <<'PY'
import math, sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = p.get("metadata") or {}
n, a = int(m.get("num_val_pairs", 0)), float(m.get("val_accuracy", float("nan")))
print(f"scorer_gate num_val={n} accuracy={a:.6f}")
if n < 40 or not math.isfinite(a) or a < 0.60:
    raise SystemExit(3)
PY

if [[ ! -s "${TRACE_REPLAY}" ]]; then
  run_stage build_trace_replay "${PYTHON}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v4.py \
    --candidate_dataset "${CANDIDATES}" --scorer_checkpoint "${SCORER}" \
    --selection scorer --select_ratio 0.25 --trajectory_length "${HORIZON}" \
    --trajectory_stride "${HORIZON}" --include_terminal_prefixes \
    --minimum_terminal_length 20 --failure_trajectory_ratio 0.20 \
    --terminal_penalty -10.0 --failure_backprop_steps 40 --failure_backprop_penalty 2.0 \
    --action_saturation_threshold 0.95 --max_saturation_fraction 0.30 \
    --action_saturation_penalty_scale 10.0 --action_delta_penalty_scale 0.10 \
    --n_step 3 --gamma 0.99 \
    --reward_source rwm_aligned --seed 42 --output "${TRACE_REPLAY}"
fi

train_branch trace_r10 "${TRACE_REPLAY}" 0.10

write_state done completed
find "${RUN_ROOT}/policies" -path '*/stage/summary.json' -print | sort > "${RUN_ROOT}/completed_branches.txt"
