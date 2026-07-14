#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${EXPERT_POLICY_PATH:?Set EXPERT_POLICY_PATH to the best 0.5-strength expert checkpoint directory.}"

VARIANT="${VARIANT:-collect_dr_final_clean}"
RUN_ID="${RUN_ID:-${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-logs/queues/go2_0p5_friend_flat_dr/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-true}"
CONFIRM_RUN="${CONFIRM_RUN:-false}"

COLLECT_GPU_DEVICE="${COLLECT_GPU_DEVICE:-cuda:0}"
WM_GPU_DEVICE="${WM_GPU_DEVICE:-cuda:0}"
SAC_GPU_DEVICE="${SAC_GPU_DEVICE:-cuda:0}"

DATASET_PATH="${DATASET_PATH:-logs/rwm_datasets/go2_rr_calf_strength_0p5_${RUN_ID}/dataset.pt}"
WM_SAVE_BASE="${WM_SAVE_BASE:-logs/rsl_rl/go2_rr_calf_strength_0p5_${RUN_ID}_wm_unmasked}"
SAC_SAVE_PATH="${SAC_SAVE_PATH:-logs/model_based/go2_rr_calf_strength_0p5_${RUN_ID}_final_unmasked/TIMESTAMP}"

COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-1024}"
COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS:-1000000}"
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-200000}"
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-5000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE:-256}"
SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS:-1024}"
SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS:-50000000}"
COLLECT_RANDOMIZATION_COMPONENTS="${COLLECT_RANDOMIZATION_COMPONENTS:-all}"
COLLECT_RANDOMIZATION_SCALE="${COLLECT_RANDOMIZATION_SCALE:-1.0}"

SKIP_COLLECT="${SKIP_COLLECT:-false}"
SKIP_WM="${SKIP_WM:-false}"
SKIP_POLICY="${SKIP_POLICY:-false}"

mkdir -p "${QUEUE_ROOT}"
SUMMARY_FILE="${QUEUE_ROOT}/summary.txt"
STATE_FILE="${QUEUE_ROOT}/state.txt"

write_state() {
  local stage="$1"
  local status="$2"
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "run_id=${RUN_ID}"
    echo "variant=${VARIANT}"
    echo "stage=${stage}"
    echo "status=${status}"
    echo "queue_root=${QUEUE_ROOT}"
  } > "${STATE_FILE}"
}

write_summary() {
  {
    echo "run_id=${RUN_ID}"
    echo "variant=${VARIANT}"
    echo "repo_root=${REPO_ROOT}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "expert_policy_path=${EXPERT_POLICY_PATH}"
    echo "dataset_path=${DATASET_PATH}"
    echo "wm_save_base=${WM_SAVE_BASE}"
    echo "sac_save_path=${SAC_SAVE_PATH}"
    echo "dry_run=${DRY_RUN}"
    echo "confirm_run=${CONFIRM_RUN}"
    echo "collect_num_transitions=${COLLECT_NUM_TRANSITIONS}"
    echo "wm_max_iterations=${WM_MAX_ITERATIONS}"
    echo "sac_num_env_steps=${SAC_NUM_ENV_STEPS}"
    echo "collect_randomization_components=${COLLECT_RANDOMIZATION_COMPONENTS}"
    echo "collect_randomization_scale=${COLLECT_RANDOMIZATION_SCALE}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${SUMMARY_FILE}"
}

run_stage() {
  local stage="$1"
  local cmd_file="$2"
  local log_file="$3"
  if [[ "${DRY_RUN}" == "true" || "${CONFIRM_RUN}" != "true" ]]; then
    write_state "${stage}" "dry_run"
    {
      echo "[dry-run] would run ${stage}"
      echo "[dry-run] command_file=${cmd_file}"
      echo "[dry-run] log_file=${log_file}"
      echo "[dry-run] set DRY_RUN=false CONFIRM_RUN=true CUDA_VISIBLE_DEVICES=<gpu> to execute"
    } | tee -a "${SUMMARY_FILE}"
    return 0
  fi
  write_state "${stage}" "running"
  echo "[run] $(date --iso-8601=seconds) ${stage}" | tee -a "${SUMMARY_FILE}"
  if bash "${cmd_file}" 2>&1 | tee "${log_file}"; then
    write_state "${stage}" "completed"
    echo "[done] $(date --iso-8601=seconds) ${stage}" | tee -a "${SUMMARY_FILE}"
  else
    write_state "${stage}" "failed"
    echo "[failed] $(date --iso-8601=seconds) ${stage}" | tee -a "${SUMMARY_FILE}"
    return 1
  fi
}

COLLECT_ENV_ACTION_NOISE_STD="0.0"
FINAL_INTERFACE_ACTION_NOISE_STD="0.0"
FINAL_INTERFACE_ACTION_BIAS_STD="0.0"
FINAL_INTERFACE_ACTION_SCALE_MIN="1.0"
FINAL_INTERFACE_ACTION_SCALE_MAX="1.0"
FINAL_INTERFACE_ACTION_DELAY_STEPS_MIN="0"
FINAL_INTERFACE_ACTION_DELAY_STEPS_MAX="0"
FINAL_INTERFACE_OBS_NOISE_PROFILE="none"
FINAL_INTERFACE_OBS_NOISE_SCALE="1.0"
FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MIN="0.0"
FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MAX="0.0"
FINAL_INTERFACE_OBS_NOISE_STD="0.0"
FINAL_INTERFACE_OBS_BIAS_STD="0.0"

case "${VARIANT}" in
  collect_dr_final_clean)
    ;;
  collect_dr_action_noise_final_clean)
    COLLECT_ENV_ACTION_NOISE_STD="${COLLECT_ENV_ACTION_NOISE_STD_OVERRIDE:-0.03}"
    ;;
  collect_dr_final_interface)
    FINAL_INTERFACE_ACTION_NOISE_STD="${FINAL_INTERFACE_ACTION_NOISE_STD_OVERRIDE:-0.02}"
    FINAL_INTERFACE_ACTION_BIAS_STD="${FINAL_INTERFACE_ACTION_BIAS_STD_OVERRIDE:-0.005}"
    FINAL_INTERFACE_ACTION_SCALE_MIN="${FINAL_INTERFACE_ACTION_SCALE_MIN_OVERRIDE:-0.95}"
    FINAL_INTERFACE_ACTION_SCALE_MAX="${FINAL_INTERFACE_ACTION_SCALE_MAX_OVERRIDE:-1.05}"
    FINAL_INTERFACE_ACTION_DELAY_STEPS_MIN="${FINAL_INTERFACE_ACTION_DELAY_STEPS_MIN_OVERRIDE:-0}"
    FINAL_INTERFACE_ACTION_DELAY_STEPS_MAX="${FINAL_INTERFACE_ACTION_DELAY_STEPS_MAX_OVERRIDE:-1}"
    FINAL_INTERFACE_OBS_NOISE_PROFILE="${FINAL_INTERFACE_OBS_NOISE_PROFILE_OVERRIDE:-legacy_friend}"
    FINAL_INTERFACE_OBS_NOISE_SCALE="${FINAL_INTERFACE_OBS_NOISE_SCALE_OVERRIDE:-1.0}"
    FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MIN="${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MIN_OVERRIDE:--0.035}"
    FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MAX="${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MAX_OVERRIDE:-0.035}"
    ;;
  collect_dr_action_noise_final_interface)
    COLLECT_ENV_ACTION_NOISE_STD="${COLLECT_ENV_ACTION_NOISE_STD_OVERRIDE:-0.03}"
    FINAL_INTERFACE_ACTION_NOISE_STD="${FINAL_INTERFACE_ACTION_NOISE_STD_OVERRIDE:-0.02}"
    FINAL_INTERFACE_ACTION_BIAS_STD="${FINAL_INTERFACE_ACTION_BIAS_STD_OVERRIDE:-0.005}"
    FINAL_INTERFACE_ACTION_SCALE_MIN="${FINAL_INTERFACE_ACTION_SCALE_MIN_OVERRIDE:-0.95}"
    FINAL_INTERFACE_ACTION_SCALE_MAX="${FINAL_INTERFACE_ACTION_SCALE_MAX_OVERRIDE:-1.05}"
    FINAL_INTERFACE_ACTION_DELAY_STEPS_MIN="${FINAL_INTERFACE_ACTION_DELAY_STEPS_MIN_OVERRIDE:-0}"
    FINAL_INTERFACE_ACTION_DELAY_STEPS_MAX="${FINAL_INTERFACE_ACTION_DELAY_STEPS_MAX_OVERRIDE:-1}"
    FINAL_INTERFACE_OBS_NOISE_PROFILE="${FINAL_INTERFACE_OBS_NOISE_PROFILE_OVERRIDE:-legacy_friend}"
    FINAL_INTERFACE_OBS_NOISE_SCALE="${FINAL_INTERFACE_OBS_NOISE_SCALE_OVERRIDE:-1.0}"
    FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MIN="${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MIN_OVERRIDE:--0.035}"
    FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MAX="${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MAX_OVERRIDE:-0.035}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    echo "Allowed: collect_dr_final_clean, collect_dr_action_noise_final_clean, collect_dr_final_interface, collect_dr_action_noise_final_interface" >&2
    exit 2
    ;;
esac

write_summary

COLLECT_CMD="${QUEUE_ROOT}/01_collect_dataset.sh"
cat > "${COLLECT_CMD}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH}" \\
DATASET_PATH="${DATASET_PATH}" \\
COLLECT_DEVICE="${COLLECT_GPU_DEVICE}" \\
COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS}" \\
COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS}" \\
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE}" \\
USE_DOMAIN_RANDOMIZATION=true \\
USE_PUSH_RANDOMIZATION=true \\
USE_OBSERVATION_NOISE=true \\
RANDOMIZATION_PRESET=friend_flat \\
RANDOMIZATION_COMPONENTS="${COLLECT_RANDOMIZATION_COMPONENTS}" \\
RANDOMIZATION_SCALE="${COLLECT_RANDOMIZATION_SCALE}" \\
ENV_ACTION_DELAY_STEPS_MIN=0 \\
ENV_ACTION_DELAY_STEPS_MAX=0 \\
ENV_ACTION_SCALE_MIN=1.0 \\
ENV_ACTION_SCALE_MAX=1.0 \\
ENV_ACTION_BIAS_STD=0.0 \\
ENV_ACTION_NOISE_STD="${COLLECT_ENV_ACTION_NOISE_STD}" \\
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/02_collect_expert_dataset.sh
EOF
chmod +x "${COLLECT_CMD}"

WM_CMD="${QUEUE_ROOT}/02_train_rwm.sh"
cat > "${WM_CMD}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
DATASET_PATH="${DATASET_PATH}" \\
WM_SAVE_BASE="${WM_SAVE_BASE}" \\
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS}" \\
WM_BATCH_SIZE="${WM_BATCH_SIZE}" \\
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE}" \\
WM_DEVICE="${WM_GPU_DEVICE}" \\
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/03_train_rwm_unmasked.sh
EOF
chmod +x "${WM_CMD}"

POLICY_CMD="${QUEUE_ROOT}/03_train_final_policy.sh"
cat > "${POLICY_CMD}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
DATASET_PATH="${DATASET_PATH}" \\
WM_SAVE_BASE="${WM_SAVE_BASE}" \\
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS}" \\
SAC_SAVE_PATH="${SAC_SAVE_PATH}" \\
SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS}" \\
SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS}" \\
SAC_DEVICE="${SAC_GPU_DEVICE}" \\
RWM_INTERFACE_ACTION_NOISE_STD="${FINAL_INTERFACE_ACTION_NOISE_STD}" \\
RWM_INTERFACE_ACTION_BIAS_STD="${FINAL_INTERFACE_ACTION_BIAS_STD}" \\
RWM_INTERFACE_ACTION_SCALE_MIN="${FINAL_INTERFACE_ACTION_SCALE_MIN}" \\
RWM_INTERFACE_ACTION_SCALE_MAX="${FINAL_INTERFACE_ACTION_SCALE_MAX}" \\
RWM_INTERFACE_ACTION_DELAY_STEPS_MIN="${FINAL_INTERFACE_ACTION_DELAY_STEPS_MIN}" \\
RWM_INTERFACE_ACTION_DELAY_STEPS_MAX="${FINAL_INTERFACE_ACTION_DELAY_STEPS_MAX}" \\
RWM_INTERFACE_OBS_NOISE_PROFILE="${FINAL_INTERFACE_OBS_NOISE_PROFILE}" \\
RWM_INTERFACE_OBS_NOISE_SCALE="${FINAL_INTERFACE_OBS_NOISE_SCALE}" \\
RWM_INTERFACE_OBS_JOINT_POS_BIAS_MIN="${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MIN}" \\
RWM_INTERFACE_OBS_JOINT_POS_BIAS_MAX="${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MAX}" \\
RWM_INTERFACE_OBS_NOISE_STD="${FINAL_INTERFACE_OBS_NOISE_STD}" \\
RWM_INTERFACE_OBS_BIAS_STD="${FINAL_INTERFACE_OBS_BIAS_STD}" \\
bash scripts/reinforcement_learning/go2_0p5_proprioceptive/05_train_policy_unmasked.sh
EOF
chmod +x "${POLICY_CMD}"

{
  echo "collect_env_action_noise_std=${COLLECT_ENV_ACTION_NOISE_STD}"
  echo "collect_randomization_components=${COLLECT_RANDOMIZATION_COMPONENTS}"
  echo "collect_randomization_scale=${COLLECT_RANDOMIZATION_SCALE}"
  echo "collect_env_action_scale_range=1.0,1.0"
  echo "collect_env_action_bias_std=0.0"
  echo "final_interface_action_noise_std=${FINAL_INTERFACE_ACTION_NOISE_STD}"
  echo "final_interface_action_bias_std=${FINAL_INTERFACE_ACTION_BIAS_STD}"
  echo "final_interface_action_scale_range=${FINAL_INTERFACE_ACTION_SCALE_MIN},${FINAL_INTERFACE_ACTION_SCALE_MAX}"
  echo "final_interface_action_delay_steps=${FINAL_INTERFACE_ACTION_DELAY_STEPS_MIN},${FINAL_INTERFACE_ACTION_DELAY_STEPS_MAX}"
  echo "final_interface_obs_noise_profile=${FINAL_INTERFACE_OBS_NOISE_PROFILE}"
  echo "final_interface_obs_noise_scale=${FINAL_INTERFACE_OBS_NOISE_SCALE}"
  echo "final_interface_obs_joint_pos_bias_range=${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MIN},${FINAL_INTERFACE_OBS_JOINT_POS_BIAS_MAX}"
  echo "final_interface_obs_noise_std=${FINAL_INTERFACE_OBS_NOISE_STD}"
  echo "command_collect=${COLLECT_CMD}"
  echo "command_wm=${WM_CMD}"
  echo "command_policy=${POLICY_CMD}"
} >> "${SUMMARY_FILE}"

if [[ "${SKIP_COLLECT}" != "true" ]]; then
  run_stage "collect_dataset" "${COLLECT_CMD}" "${QUEUE_ROOT}/01_collect_dataset.log"
elif [[ "${DRY_RUN}" != "true" && ! -f "${DATASET_PATH}" ]]; then
  echo "SKIP_COLLECT=true but dataset missing: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_WM}" != "true" ]]; then
  run_stage "train_rwm" "${WM_CMD}" "${QUEUE_ROOT}/02_train_rwm.log"
fi

if [[ "${SKIP_POLICY}" != "true" ]]; then
  run_stage "train_final_policy" "${POLICY_CMD}" "${QUEUE_ROOT}/03_train_final_policy.log"
fi

if [[ "${DRY_RUN}" == "true" || "${CONFIRM_RUN}" != "true" ]]; then
  write_state "pipeline" "dry_run"
else
  write_state "pipeline" "completed"
fi
