#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
elif [[ -f "${SCRIPT_DIR}/pyproject.toml" && -d "${SCRIPT_DIR}/scripts" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

SEED="${SEED:-0}"
REUSE_EXISTING="${REUSE_EXISTING:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
DRY_RUN="${DRY_RUN:-true}"
CONFIRM_RUN="${CONFIRM_RUN:-false}"
ALLOW_ALL_GPUS="${ALLOW_ALL_GPUS:-false}"

QUEUE_ROOT="${QUEUE_ROOT:-logs/queues/go2_proprioceptive_rwm_sac_pipeline/$(date +%Y-%m-%d_%H-%M-%S)}"
STATE_FILE="${QUEUE_ROOT}/state.txt"
SUMMARY_FILE="${QUEUE_ROOT}/summary.txt"

PRETRAIN_TASK="${PRETRAIN_TASK:-Unitree-Go2-Flat-RWM-Pretrain-Ens}"
PRETRAIN_SAVE_BASE="${PRETRAIN_SAVE_BASE:-logs/rsl_rl/go2_flat_rwm_flashsac_pretrain}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-}"
PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-}"
PRETRAIN_DEVICE="${PRETRAIN_DEVICE:-cuda:0}"
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-512}"
PRETRAIN_NUM_ENV_STEPS="${PRETRAIN_NUM_ENV_STEPS:-100000000}"
PRETRAIN_SAVE_INTERVAL="${PRETRAIN_SAVE_INTERVAL:-1000}"
PRETRAIN_UPDATES_PER_STEP="${PRETRAIN_UPDATES_PER_STEP:-1}"
PRETRAIN_AGENT_BUFFER_MAX_LENGTH="${PRETRAIN_AGENT_BUFFER_MAX_LENGTH:-10000000}"
PRETRAIN_AGENT_BUFFER_MIN_LENGTH="${PRETRAIN_AGENT_BUFFER_MIN_LENGTH:-50000}"
PRETRAIN_AGENT_BUFFER_DEVICE_TYPE="${PRETRAIN_AGENT_BUFFER_DEVICE_TYPE:-cpu}"
PRETRAIN_AGENT_SAMPLE_BATCH_SIZE="${PRETRAIN_AGENT_SAMPLE_BATCH_SIZE:-2048}"
PRETRAIN_AGENT_N_STEP="${PRETRAIN_AGENT_N_STEP:-3}"
PRETRAIN_DYN_BATCH_SIZE="${PRETRAIN_DYN_BATCH_SIZE:-256}"
PRETRAIN_DYN_MIN_TRANSITIONS="${PRETRAIN_DYN_MIN_TRANSITIONS:-50000}"
PRETRAIN_DYN_UPDATES_PER_STEP="${PRETRAIN_DYN_UPDATES_PER_STEP:-1}"
PRETRAIN_USE_DOMAIN_RANDOMIZATION="${PRETRAIN_USE_DOMAIN_RANDOMIZATION:-true}"
PRETRAIN_USE_PUSH_RANDOMIZATION="${PRETRAIN_USE_PUSH_RANDOMIZATION:-true}"
PRETRAIN_USE_OBSERVATION_NOISE="${PRETRAIN_USE_OBSERVATION_NOISE:-true}"

EXPERT_SAVE_BASE="${EXPERT_SAVE_BASE:-logs/model_based/go2_flat_flashsac_rwm_sacwm}"
EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH:-}"
EXPERT_DEVICE="${EXPERT_DEVICE:-cuda:0}"
EXPERT_NUM_IMAGINATION_ENVS="${EXPERT_NUM_IMAGINATION_ENVS:-1024}"
EXPERT_NUM_ENV_STEPS="${EXPERT_NUM_ENV_STEPS:-50000000}"
EXPERT_SAVE_INTERVAL="${EXPERT_SAVE_INTERVAL:-1000}"
EXPERT_UPDATES_PER_STEP="${EXPERT_UPDATES_PER_STEP:-2}"
EXPERT_BUFFER_MAX_LENGTH="${EXPERT_BUFFER_MAX_LENGTH:-10000000}"
EXPERT_BUFFER_MIN_LENGTH="${EXPERT_BUFFER_MIN_LENGTH:-100000}"
EXPERT_BUFFER_DEVICE_TYPE="${EXPERT_BUFFER_DEVICE_TYPE:-cpu}"
EXPERT_SAMPLE_BATCH_SIZE="${EXPERT_SAMPLE_BATCH_SIZE:-2048}"
EXPERT_N_STEP="${EXPERT_N_STEP:-3}"
EXPERT_NORMALIZE_REWARD="${EXPERT_NORMALIZE_REWARD:-true}"
EXPERT_NORMALIZED_G_MAX="${EXPERT_NORMALIZED_G_MAX:-5.0}"
EXPERT_CRITIC_MIN_V="${EXPERT_CRITIC_MIN_V:--5.0}"
EXPERT_CRITIC_MAX_V="${EXPERT_CRITIC_MAX_V:-5.0}"
EXPERT_UNCERTAINTY_PENALTY_WEIGHT="${EXPERT_UNCERTAINTY_PENALTY_WEIGHT:--1.0}"

DATASET_PATH="${DATASET_PATH:-logs/rwm_datasets/go2_flat_mixed_safety_command_coverage_1m/dataset.pt}"
MEDIUM_POLICY_PATH="${MEDIUM_POLICY_PATH:-}"
COLLECT_TASK="${COLLECT_TASK:-Unitree-Go2-Flat-RWM-Pretrain-Ens}"
COLLECT_DEVICE="${COLLECT_DEVICE:-cuda:0}"
COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-1024}"
COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS:-1000000}"
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-200000}"
COLLECTOR_MIX="${COLLECTOR_MIX:-expert:0.45,noisy_expert:0.25,medium:0.10,failure_border:0.15,random:0.05}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.12}"
MEDIUM_ACTION_NOISE_STD="${MEDIUM_ACTION_NOISE_STD:-0.25}"
FAILURE_ACTION_NOISE_STD="${FAILURE_ACTION_NOISE_STD:-0.45}"
COMMAND_MODES="${COMMAND_MODES:-stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw}"
COMMAND_MODE_WEIGHTS="${COMMAND_MODE_WEIGHTS:-stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13}"
X_ABS_RANGE_MIN="${X_ABS_RANGE_MIN:-0.05}"
X_ABS_RANGE_MAX="${X_ABS_RANGE_MAX:-0.5}"
Y_ABS_RANGE_MIN="${Y_ABS_RANGE_MIN:-0.03}"
Y_ABS_RANGE_MAX="${Y_ABS_RANGE_MAX:-0.2}"
YAW_ABS_RANGE_MIN="${YAW_ABS_RANGE_MIN:-0.05}"
YAW_ABS_RANGE_MAX="${YAW_ABS_RANGE_MAX:-0.4}"
COMMAND_RESAMPLE_INTERVAL_MIN="${COMMAND_RESAMPLE_INTERVAL_MIN:-120}"
COMMAND_RESAMPLE_INTERVAL_MAX="${COMMAND_RESAMPLE_INTERVAL_MAX:-300}"

WM_SAVE_BASE="${WM_SAVE_BASE:-logs/rsl_rl/go2_flat_rwm_offline_mixed_safety_command_coverage_1m_proprioceptive}"
WM_MODEL_PATH="${WM_MODEL_PATH:-}"
WM_DEVICE="${WM_DEVICE:-cuda:0}"
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-5000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE:-256}"
WM_SAVE_INTERVAL="${WM_SAVE_INTERVAL:-500}"
WM_LOG_INTERVAL="${WM_LOG_INTERVAL:-50}"

SAC_SAVE_BASE="${SAC_SAVE_BASE:-logs/model_based/go2_flat_flashsac_rwm_proprioceptive_1m}"
SAC_POLICY_PATH="${SAC_POLICY_PATH:-}"
SAC_DEVICE="${SAC_DEVICE:-cuda:0}"
SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS:-1024}"
SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS:-50000000}"
SAC_SAVE_INTERVAL="${SAC_SAVE_INTERVAL:-1000}"
SAC_UPDATES_PER_STEP="${SAC_UPDATES_PER_STEP:-2}"
SAC_BUFFER_MAX_LENGTH="${SAC_BUFFER_MAX_LENGTH:-10000000}"
SAC_BUFFER_MIN_LENGTH="${SAC_BUFFER_MIN_LENGTH:-100000}"
SAC_BUFFER_DEVICE_TYPE="${SAC_BUFFER_DEVICE_TYPE:-cpu}"
SAC_SAMPLE_BATCH_SIZE="${SAC_SAMPLE_BATCH_SIZE:-2048}"
SAC_N_STEP="${SAC_N_STEP:-3}"
SAC_NORMALIZE_REWARD="${SAC_NORMALIZE_REWARD:-true}"
SAC_NORMALIZED_G_MAX="${SAC_NORMALIZED_G_MAX:-5.0}"
SAC_CRITIC_MIN_V="${SAC_CRITIC_MIN_V:--5.0}"
SAC_CRITIC_MAX_V="${SAC_CRITIC_MAX_V:-5.0}"
SAC_UNCERTAINTY_PENALTY_WEIGHT="${SAC_UNCERTAINTY_PENALTY_WEIGHT:--2.0}"
SAC_LIN_VEL_X_MIN="${SAC_LIN_VEL_X_MIN:--0.3}"
SAC_LIN_VEL_X_MAX="${SAC_LIN_VEL_X_MAX:-0.3}"
SAC_LIN_VEL_Y_MIN="${SAC_LIN_VEL_Y_MIN:--0.15}"
SAC_LIN_VEL_Y_MAX="${SAC_LIN_VEL_Y_MAX:-0.15}"
SAC_ANG_VEL_Z_MIN="${SAC_ANG_VEL_Z_MIN:--0.3}"
SAC_ANG_VEL_Z_MAX="${SAC_ANG_VEL_Z_MAX:-0.3}"

EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1024}"
EVAL_STEPS="${EVAL_STEPS:-1001}"
EVAL_FIXED_COMMAND_X="${EVAL_FIXED_COMMAND_X:-0.3}"
EVAL_FIXED_COMMAND_Y="${EVAL_FIXED_COMMAND_Y:-0.0}"
EVAL_FIXED_COMMAND_YAW="${EVAL_FIXED_COMMAND_YAW:-0.0}"
EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-logs/evals/go2_proprioceptive_$(date +%Y-%m-%d_%H-%M-%S).json}"

mkdir -p "${QUEUE_ROOT}" "$(dirname "${DATASET_PATH}")" "${WM_SAVE_BASE}" "${SAC_SAVE_BASE}" "$(dirname "${EVAL_OUTPUT_JSON}")"

write_state() {
  local stage="$1"
  local status="$2"
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "stage=${stage}"
    echo "status=${status}"
    echo "queue_root=${QUEUE_ROOT}"
  } > "${STATE_FILE}"
}

append_summary() {
  {
    echo "$@"
  } >> "${SUMMARY_FILE}"
}

run_stage() {
  local stage="$1"
  local cmd_file="$2"
  local log_file="$3"
  if [[ "${DRY_RUN}" == "true" ]]; then
    write_state "${stage}" "dry_run"
    append_summary "[dry-run] would run ${stage}"
    append_summary "[dry-run] command_file=${cmd_file}"
    append_summary "[dry-run] log_file=${log_file}"
    append_summary "[dry-run] set DRY_RUN=false CONFIRM_RUN=true CUDA_VISIBLE_DEVICES=<gpu> to execute"
    echo "[dry-run] generated ${cmd_file}"
    echo "[dry-run] not executing ${stage}"
    exit 0
  fi
  write_state "${stage}" "running"
  append_summary "[queue] $(date --iso-8601=seconds) starting ${stage}"
  if bash "${cmd_file}" > "${log_file}" 2>&1; then
    append_summary "[queue] $(date --iso-8601=seconds) completed ${stage}"
    write_state "${stage}" "completed"
  else
    local exit_code="$?"
    append_summary "[queue] $(date --iso-8601=seconds) failed ${stage} exit_code=${exit_code}"
    append_summary "[queue] see log_file=${log_file}"
    write_state "${stage}" "failed"
    return "${exit_code}"
  fi
}

latest_child_dir() {
  local base="$1"
  [[ -d "${base}" ]] || return 1
  find "${base}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_model_file() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  find "${dir}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f %p\n' \
    | sed -E 's/^model_([0-9]+)\.pt /\1 /' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

latest_step_dir() {
  local base="$1"
  local config_name="${2:-rwm_flashsac_config.yaml}"
  [[ -d "${base}" ]] || return 1
  while IFS= read -r dir; do
    if [[ -f "${dir}/${config_name}" ]]; then
      echo "${dir}"
      return 0
    fi
  done < <(find "${base}" -mindepth 3 -maxdepth 3 -type f -name 'actor.pt' -printf '%T@ %h\n' \
    | sort -nr \
    | cut -d' ' -f2-)
  return 1
}

latest_pretrain_pair() {
  local base="$1"
  [[ -d "${base}" ]] || return 1
  while IFS= read -r run_dir; do
    local model_path
    model_path="$(latest_model_file "${run_dir}" || true)"
    if [[ -n "${model_path}" && -f "${run_dir}/dataset.pt" ]]; then
      echo "${model_path}|${run_dir}/dataset.pt"
      return 0
    fi
  done < <(find "${base}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | cut -d' ' -f2-)
  return 1
}

latest_wm_model_file() {
  local base="$1"
  local target_iteration="$2"
  [[ -d "${base}" ]] || return 1
  while IFS= read -r run_dir; do
    if [[ -f "${run_dir}/model_${target_iteration}.pt" ]]; then
      echo "${run_dir}/model_${target_iteration}.pt"
      return 0
    fi
    if [[ -f "${run_dir}/latest.pt" ]]; then
      echo "${run_dir}/latest.pt"
      return 0
    fi
  done < <(find "${base}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | cut -d' ' -f2-)
  return 1
}

write_summary_header() {
  {
    echo "queue_root=${QUEUE_ROOT}"
    echo "repo_root=${REPO_ROOT}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "seed=${SEED}"
    echo "reuse_existing=${REUSE_EXISTING}"
    echo "run_eval=${RUN_EVAL}"
    echo "dry_run=${DRY_RUN}"
    echo "confirm_run=${CONFIRM_RUN}"
    echo "allow_all_gpus=${ALLOW_ALL_GPUS}"
    echo "pretrain_save_base=${PRETRAIN_SAVE_BASE}"
    echo "expert_save_base=${EXPERT_SAVE_BASE}"
    echo "dataset_path=${DATASET_PATH}"
    echo "wm_save_base=${WM_SAVE_BASE}"
    echo "sac_save_base=${SAC_SAVE_BASE}"
    echo "eval_output_json=${EVAL_OUTPUT_JSON}"
    echo "started_at=$(date --iso-8601=seconds)"
    echo
  } > "${SUMMARY_FILE}"
}

write_summary_header
write_state "precheck" "running"

if [[ "${DRY_RUN}" != "true" && "${CONFIRM_RUN}" != "true" ]]; then
  write_state "precheck" "blocked"
  append_summary "[queue] blocked: set CONFIRM_RUN=true to execute non-dry-run stages"
  echo "Blocked: set CONFIRM_RUN=true to execute. Default DRY_RUN=true only generates the next command."
  exit 2
fi

if [[ "${DRY_RUN}" != "true" && -z "${CUDA_VISIBLE_DEVICES}" && "${ALLOW_ALL_GPUS}" != "true" ]]; then
  write_state "precheck" "blocked"
  append_summary "[queue] blocked: CUDA_VISIBLE_DEVICES is empty"
  echo "Blocked: set CUDA_VISIBLE_DEVICES=<gpu> or ALLOW_ALL_GPUS=true."
  exit 2
fi

if [[ "${REUSE_EXISTING}" == "true" && -z "${EXPERT_POLICY_PATH}" ]]; then
  EXPERT_POLICY_PATH="$(latest_step_dir "${EXPERT_SAVE_BASE}" "rwm_flashsac_config.yaml" || true)"
  if [[ -n "${EXPERT_POLICY_PATH}" ]]; then
    append_summary "[queue] reusing expert policy: ${EXPERT_POLICY_PATH}"
  fi
fi

if [[ -z "${EXPERT_POLICY_PATH}" ]]; then
  if [[ "${REUSE_EXISTING}" == "true" && ( -z "${PRETRAIN_MODEL_PATH}" || -z "${PRETRAIN_DATASET_PATH}" ) ]]; then
    PRETRAIN_PAIR="$(latest_pretrain_pair "${PRETRAIN_SAVE_BASE}" || true)"
    if [[ -n "${PRETRAIN_PAIR}" ]]; then
      PRETRAIN_MODEL_PATH="${PRETRAIN_PAIR%%|*}"
      PRETRAIN_DATASET_PATH="${PRETRAIN_PAIR#*|}"
    fi
  fi

  if [[ -z "${PRETRAIN_MODEL_PATH}" || -z "${PRETRAIN_DATASET_PATH}" ]]; then
    cat > "${QUEUE_ROOT}/01_pretrain_flashsac_rwm_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export MUJOCO_GL="${MUJOCO_GL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
uv run python scripts/reinforcement_learning/rwm_flashsac/train_pretrain_flashsac_go2.py \\
  --config_path scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm_pretrain.yaml \\
  --task "${PRETRAIN_TASK}" \\
  --device "${PRETRAIN_DEVICE}" \\
  --seed "${SEED}" \\
  --num_envs "${PRETRAIN_NUM_ENVS}" \\
  --num_env_steps "${PRETRAIN_NUM_ENV_STEPS}" \\
  --save_path "${PRETRAIN_SAVE_BASE}/TIMESTAMP" \\
  --overrides updates_per_interaction_step="${PRETRAIN_UPDATES_PER_STEP}" \\
  --overrides save_checkpoint_per_interaction_step="${PRETRAIN_SAVE_INTERVAL}" \\
  --overrides save_dataset=true \\
  --overrides env.use_domain_randomization="${PRETRAIN_USE_DOMAIN_RANDOMIZATION}" \\
  --overrides env.use_push_randomization="${PRETRAIN_USE_PUSH_RANDOMIZATION}" \\
  --overrides env.use_observation_noise="${PRETRAIN_USE_OBSERVATION_NOISE}" \\
  --overrides agent.buffer_max_length="${PRETRAIN_AGENT_BUFFER_MAX_LENGTH}" \\
  --overrides agent.buffer_min_length="${PRETRAIN_AGENT_BUFFER_MIN_LENGTH}" \\
  --overrides agent.buffer_device_type="${PRETRAIN_AGENT_BUFFER_DEVICE_TYPE}" \\
  --overrides agent.sample_batch_size="${PRETRAIN_AGENT_SAMPLE_BATCH_SIZE}" \\
  --overrides agent.n_step="${PRETRAIN_AGENT_N_STEP}" \\
  --overrides system_dynamics.batch_size="${PRETRAIN_DYN_BATCH_SIZE}" \\
  --overrides system_dynamics.min_transitions="${PRETRAIN_DYN_MIN_TRANSITIONS}" \\
  --overrides system_dynamics.updates_per_interaction_step="${PRETRAIN_DYN_UPDATES_PER_STEP}"
EOF
    chmod +x "${QUEUE_ROOT}/01_pretrain_flashsac_rwm_command.sh"
    run_stage "pretrain_flashsac_rwm" "${QUEUE_ROOT}/01_pretrain_flashsac_rwm_command.sh" "${QUEUE_ROOT}/01_pretrain_flashsac_rwm.log"

    PRETRAIN_RUN_DIR="$(latest_child_dir "${PRETRAIN_SAVE_BASE}")"
    PRETRAIN_MODEL_PATH="$(latest_model_file "${PRETRAIN_RUN_DIR}")"
    PRETRAIN_DATASET_PATH="${PRETRAIN_RUN_DIR}/dataset.pt"
  else
    append_summary "[queue] reusing pretrain model: ${PRETRAIN_MODEL_PATH}"
    append_summary "[queue] reusing pretrain dataset: ${PRETRAIN_DATASET_PATH}"
  fi

  if [[ ! -f "${PRETRAIN_MODEL_PATH}" ]]; then
    write_state "find_pretrain_model" "failed"
    append_summary "[queue] missing pretrain model: ${PRETRAIN_MODEL_PATH}"
    exit 1
  fi
  if [[ ! -f "${PRETRAIN_DATASET_PATH}" ]]; then
    write_state "find_pretrain_dataset" "failed"
    append_summary "[queue] missing pretrain dataset: ${PRETRAIN_DATASET_PATH}"
    exit 1
  fi

  cat > "${QUEUE_ROOT}/02_train_expert_policy_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export MUJOCO_GL="${MUJOCO_GL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2.py \\
  --config_path scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm.yaml \\
  --model_resume_path "${PRETRAIN_MODEL_PATH}" \\
  --dataset_path "${PRETRAIN_DATASET_PATH}" \\
  --device "${EXPERT_DEVICE}" \\
  --num_imagination_envs "${EXPERT_NUM_IMAGINATION_ENVS}" \\
  --num_env_steps "${EXPERT_NUM_ENV_STEPS}" \\
  --save_path "${EXPERT_SAVE_BASE}/TIMESTAMP" \\
  --overrides updates_per_interaction_step="${EXPERT_UPDATES_PER_STEP}" \\
  --overrides save_checkpoint_per_interaction_step="${EXPERT_SAVE_INTERVAL}" \\
  --overrides world_model.uncertainty_penalty_weight="${EXPERT_UNCERTAINTY_PENALTY_WEIGHT}" \\
  --overrides agent.buffer_max_length="${EXPERT_BUFFER_MAX_LENGTH}" \\
  --overrides agent.buffer_min_length="${EXPERT_BUFFER_MIN_LENGTH}" \\
  --overrides agent.buffer_device_type="${EXPERT_BUFFER_DEVICE_TYPE}" \\
  --overrides agent.sample_batch_size="${EXPERT_SAMPLE_BATCH_SIZE}" \\
  --overrides agent.n_step="${EXPERT_N_STEP}" \\
  --overrides agent.normalize_reward="${EXPERT_NORMALIZE_REWARD}" \\
  --overrides agent.normalized_G_max="${EXPERT_NORMALIZED_G_MAX}" \\
  --overrides agent.critic_min_v="${EXPERT_CRITIC_MIN_V}" \\
  --overrides agent.critic_max_v="${EXPERT_CRITIC_MAX_V}"
EOF
  chmod +x "${QUEUE_ROOT}/02_train_expert_policy_command.sh"
  run_stage "train_expert_policy" "${QUEUE_ROOT}/02_train_expert_policy_command.sh" "${QUEUE_ROOT}/02_train_expert_policy.log"
  EXPERT_POLICY_PATH="$(latest_step_dir "${EXPERT_SAVE_BASE}")"
fi

if [[ -z "${EXPERT_POLICY_PATH}" || ! -f "${EXPERT_POLICY_PATH}/actor.pt" ]]; then
  write_state "find_expert_policy" "failed"
  append_summary "[queue] missing expert policy directory or actor.pt: ${EXPERT_POLICY_PATH}"
  exit 1
fi
append_summary "expert_policy_path=${EXPERT_POLICY_PATH}"

if [[ "${REUSE_EXISTING}" == "true" && -f "${DATASET_PATH}" ]]; then
  append_summary "[queue] reusing dataset: ${DATASET_PATH}"
else
  cat > "${QUEUE_ROOT}/03_collect_mixed_dataset_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export MUJOCO_GL="${MUJOCO_GL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
rm -rf "$(dirname "${DATASET_PATH}")/parts"
uv run python scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \\
  --task "${COLLECT_TASK}" \\
  --device "${COLLECT_DEVICE}" \\
  --seed "${SEED}" \\
  --num_envs "${COLLECT_NUM_ENVS}" \\
  --num_transitions "${COLLECT_NUM_TRANSITIONS}" \\
  --save_path "${DATASET_PATH}" \\
  --expert_policy_path "${EXPERT_POLICY_PATH}" \\
  --medium_policy_path "${MEDIUM_POLICY_PATH}" \\
  --collector_mix "${COLLECTOR_MIX}" \\
  --action_noise_std "${ACTION_NOISE_STD}" \\
  --medium_action_noise_std "${MEDIUM_ACTION_NOISE_STD}" \\
  --failure_action_noise_std "${FAILURE_ACTION_NOISE_STD}" \\
  --chunk_size "${COLLECT_CHUNK_SIZE}" \\
  --command_modes "${COMMAND_MODES}" \\
  --command_mode_weights "${COMMAND_MODE_WEIGHTS}" \\
  --x_range "-${X_ABS_RANGE_MAX}" "${X_ABS_RANGE_MAX}" \\
  --signed_x \\
  --x_abs_range "${X_ABS_RANGE_MIN}" "${X_ABS_RANGE_MAX}" \\
  --y_abs_range "${Y_ABS_RANGE_MIN}" "${Y_ABS_RANGE_MAX}" \\
  --yaw_abs_range "${YAW_ABS_RANGE_MIN}" "${YAW_ABS_RANGE_MAX}" \\
  --command_resample_interval_min "${COMMAND_RESAMPLE_INTERVAL_MIN}" \\
  --command_resample_interval_max "${COMMAND_RESAMPLE_INTERVAL_MAX}" \\
  --no-use_domain_randomization \\
  --no-use_push_randomization \\
  --no-use_observation_noise \\
  --headless
EOF
  chmod +x "${QUEUE_ROOT}/03_collect_mixed_dataset_command.sh"
  run_stage "collect_mixed_dataset" "${QUEUE_ROOT}/03_collect_mixed_dataset_command.sh" "${QUEUE_ROOT}/03_collect_mixed_dataset.log"
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  write_state "find_dataset" "failed"
  append_summary "[queue] missing dataset: ${DATASET_PATH}"
  exit 1
fi
append_summary "dataset_path=${DATASET_PATH}"

  if [[ "${REUSE_EXISTING}" == "true" && -z "${WM_MODEL_PATH}" ]]; then
  WM_MODEL_PATH="$(latest_wm_model_file "${WM_SAVE_BASE}" "${WM_MAX_ITERATIONS}" || true)"
fi

if [[ -z "${WM_MODEL_PATH}" ]]; then
  cat > "${QUEUE_ROOT}/04_train_proprioceptive_wm_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export MUJOCO_GL="${MUJOCO_GL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
uv run python scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2_proprioceptive.py \\
  --config_path scripts/reinforcement_learning/rwm_dataset/configs/go2_offline_world_model_proprioceptive.yaml \\
  --dataset_path "${DATASET_PATH}" \\
  --save_dir "${WM_SAVE_BASE}" \\
  --max_iterations "${WM_MAX_ITERATIONS}" \\
  --batch_size "${WM_BATCH_SIZE}" \\
  --micro_batch_size "${WM_MICRO_BATCH_SIZE}" \\
  --save_interval "${WM_SAVE_INTERVAL}" \\
  --log_interval "${WM_LOG_INTERVAL}" \\
  --device "${WM_DEVICE}"
EOF
  chmod +x "${QUEUE_ROOT}/04_train_proprioceptive_wm_command.sh"
  run_stage "train_proprioceptive_world_model" "${QUEUE_ROOT}/04_train_proprioceptive_wm_command.sh" "${QUEUE_ROOT}/04_train_proprioceptive_wm.log"

  WM_RUN_DIR="$(latest_child_dir "${WM_SAVE_BASE}")"
  if [[ -f "${WM_RUN_DIR}/model_${WM_MAX_ITERATIONS}.pt" ]]; then
    WM_MODEL_PATH="${WM_RUN_DIR}/model_${WM_MAX_ITERATIONS}.pt"
  else
    WM_MODEL_PATH="${WM_RUN_DIR}/latest.pt"
  fi
else
  append_summary "[queue] reusing proprioceptive world model: ${WM_MODEL_PATH}"
fi

if [[ ! -f "${WM_MODEL_PATH}" ]]; then
  write_state "find_proprioceptive_world_model" "failed"
  append_summary "[queue] missing proprioceptive world model: ${WM_MODEL_PATH}"
  exit 1
fi
append_summary "wm_model_path=${WM_MODEL_PATH}"

if [[ "${REUSE_EXISTING}" == "true" && -z "${SAC_POLICY_PATH}" ]]; then
  SAC_POLICY_PATH="$(latest_step_dir "${SAC_SAVE_BASE}" "rwm_flashsac_config.yaml" || true)"
fi

if [[ -z "${SAC_POLICY_PATH}" ]]; then
  cat > "${QUEUE_ROOT}/05_train_proprioceptive_sac_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export MUJOCO_GL="${MUJOCO_GL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_proprioceptive.py \\
  --config_path scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm_proprioceptive_200k.yaml \\
  --model_resume_path "${WM_MODEL_PATH}" \\
  --dataset_path "${DATASET_PATH}" \\
  --device "${SAC_DEVICE}" \\
  --num_imagination_envs "${SAC_NUM_IMAGINATION_ENVS}" \\
  --num_env_steps "${SAC_NUM_ENV_STEPS}" \\
  --save_path "${SAC_SAVE_BASE}/TIMESTAMP" \\
  --overrides updates_per_interaction_step="${SAC_UPDATES_PER_STEP}" \\
  --overrides save_checkpoint_per_interaction_step="${SAC_SAVE_INTERVAL}" \\
  --overrides world_model.uncertainty_penalty_weight="${SAC_UNCERTAINTY_PENALTY_WEIGHT}" \\
  --overrides world_model.lin_vel_x_min="${SAC_LIN_VEL_X_MIN}" \\
  --overrides world_model.lin_vel_x_max="${SAC_LIN_VEL_X_MAX}" \\
  --overrides world_model.lin_vel_y_min="${SAC_LIN_VEL_Y_MIN}" \\
  --overrides world_model.lin_vel_y_max="${SAC_LIN_VEL_Y_MAX}" \\
  --overrides world_model.ang_vel_z_min="${SAC_ANG_VEL_Z_MIN}" \\
  --overrides world_model.ang_vel_z_max="${SAC_ANG_VEL_Z_MAX}" \\
  --overrides agent.buffer_max_length="${SAC_BUFFER_MAX_LENGTH}" \\
  --overrides agent.buffer_min_length="${SAC_BUFFER_MIN_LENGTH}" \\
  --overrides agent.buffer_device_type="${SAC_BUFFER_DEVICE_TYPE}" \\
  --overrides agent.sample_batch_size="${SAC_SAMPLE_BATCH_SIZE}" \\
  --overrides agent.n_step="${SAC_N_STEP}" \\
  --overrides agent.normalize_reward="${SAC_NORMALIZE_REWARD}" \\
  --overrides agent.normalized_G_max="${SAC_NORMALIZED_G_MAX}" \\
  --overrides agent.critic_min_v="${SAC_CRITIC_MIN_V}" \\
  --overrides agent.critic_max_v="${SAC_CRITIC_MAX_V}"
EOF
  chmod +x "${QUEUE_ROOT}/05_train_proprioceptive_sac_command.sh"
  run_stage "train_proprioceptive_sac" "${QUEUE_ROOT}/05_train_proprioceptive_sac_command.sh" "${QUEUE_ROOT}/05_train_proprioceptive_sac.log"
  SAC_POLICY_PATH="$(latest_step_dir "${SAC_SAVE_BASE}")"
else
  append_summary "[queue] reusing proprioceptive SAC policy: ${SAC_POLICY_PATH}"
fi

if [[ -z "${SAC_POLICY_PATH}" || ! -f "${SAC_POLICY_PATH}/actor.pt" ]]; then
  write_state "find_proprioceptive_sac_policy" "failed"
  append_summary "[queue] missing proprioceptive SAC policy directory or actor.pt: ${SAC_POLICY_PATH}"
  exit 1
fi
append_summary "sac_policy_path=${SAC_POLICY_PATH}"

if [[ "${RUN_EVAL}" == "true" ]]; then
  cat > "${QUEUE_ROOT}/06_eval_proprioceptive_policy_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export MUJOCO_GL="${MUJOCO_GL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
uv run python scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py \\
  --checkpoint_path "${SAC_POLICY_PATH}" \\
  --task "${COLLECT_TASK}" \\
  --device "${EVAL_DEVICE}" \\
  --num_envs "${EVAL_NUM_ENVS}" \\
  --steps "${EVAL_STEPS}" \\
  --seed "${SEED}" \\
  --fixed_command "${EVAL_FIXED_COMMAND_X}" "${EVAL_FIXED_COMMAND_Y}" "${EVAL_FIXED_COMMAND_YAW}" \\
  --output_json "${EVAL_OUTPUT_JSON}"
EOF
  chmod +x "${QUEUE_ROOT}/06_eval_proprioceptive_policy_command.sh"
  run_stage "eval_proprioceptive_policy" "${QUEUE_ROOT}/06_eval_proprioceptive_policy_command.sh" "${QUEUE_ROOT}/06_eval_proprioceptive_policy.log"
  append_summary "eval_output_json=${EVAL_OUTPUT_JSON}"
fi

write_state "pipeline" "completed"
append_summary "[queue] $(date --iso-8601=seconds) pipeline completed"
