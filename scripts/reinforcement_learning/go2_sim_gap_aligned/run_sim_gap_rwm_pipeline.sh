#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${GAP_ID:?Set GAP_ID}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH:-${REPO_ROOT}/logs/experiments/go2_gap_experts/selected_g0d0/step97656}"
RAW_DIR="${RUN_ROOT}/raw_1m"
SUBSET_PATH="${RUN_ROOT}/dataset_expert_command_25k.pt"
RWM_DIR="${RUN_ROOT}/rwm_baseline"
POLICY_DIR="${RUN_ROOT}/final_policy_rwm_p0"
STATE_FILE="${RUN_ROOT}/state.txt"
QUEUE_LOG="${RUN_ROOT}/run.log"

PAYLOAD_KG="${PAYLOAD_KG:-0.0}"
RR_STRENGTH="${RR_STRENGTH:-1.0}"
NUM_ENVS=1000
SHARD_TRANSITIONS=250000
COLLECTOR_MIX="expert:0.45,noisy_expert:0.25,medium:0.10,failure_border:0.15,random:0.05"
DR_COMPONENTS="friction,mass_com,motor,delay,observation,push,initial_state"

mkdir -p "${RUN_ROOT}" "${RAW_DIR}/shards"

write_state() {
  local stage="$1" status="$2"
  printf 'time=%s\ngap_id=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "${GAP_ID}" "${stage}" "${status}" "${CUDA_VISIBLE_DEVICES}" \
    > "${STATE_FILE}"
}

on_exit() {
  local code=$?
  if (( code != 0 )); then
    write_state "${CURRENT_STAGE:-unknown}" failed
  fi
  exit "${code}"
}
trap on_exit EXIT

run_stage() {
  CURRENT_STAGE="$1"
  shift
  write_state "${CURRENT_STAGE}" running
  echo "[$(date --iso-8601=seconds)] start ${CURRENT_STAGE}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee -a "${QUEUE_LOG}"
  local code=${PIPESTATUS[0]}
  if (( code != 0 )); then
    return "${code}"
  fi
  echo "[$(date --iso-8601=seconds)] complete ${CURRENT_STAGE}" | tee -a "${QUEUE_LOG}"
}

if [[ ! -f "${EXPERT_POLICY_PATH}/actor.pt" ]]; then
  echo "Missing G0_D0 expert: ${EXPERT_POLICY_PATH}/actor.pt" >&2
  exit 2
fi

SHARDS=()
for shard_id in 0 1 2 3; do
  shard_path="${RAW_DIR}/shards/shard_${shard_id}/dataset.pt"
  SHARDS+=("${shard_path}")
  if [[ -f "${shard_path}" ]]; then
    echo "[reuse] ${shard_path}" | tee -a "${QUEUE_LOG}"
    continue
  fi
  mkdir -p "$(dirname "${shard_path}")"
  run_stage "collect_shard_${shard_id}" "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
    --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
    --device cuda:0 \
    --seed "$((100 + shard_id))" \
    --shard_id "${shard_id}" \
    --num_envs "${NUM_ENVS}" \
    --num_transitions "${SHARD_TRANSITIONS}" \
    --save_path "${shard_path}" \
    --expert_policy_path "${EXPERT_POLICY_PATH}" \
    --medium_policy_path "${EXPERT_POLICY_PATH}" \
    --collector_mix "${COLLECTOR_MIX}" \
    --dataset_obs_kind proprioceptive \
    --action_noise_std 0.12 \
    --medium_action_noise_std 0.25 \
    --failure_action_noise_std 0.45 \
    --randomization_preset calibrated_default \
    --randomization_components "${DR_COMPONENTS}" \
    --randomization_scale 1.0 \
    --payload_mass_range_kg "${PAYLOAD_KG}" "${PAYLOAD_KG}" \
    --payload_position_body_m 0.0 0.0 0.10 \
    --payload_box_size_m 0.20 0.12 0.05 \
    --rr_calf_strength_range "${RR_STRENGTH}" "${RR_STRENGTH}" \
    --fixed_collector_assignment \
    --collector_assignment_seed 42 \
    --env_action_noise_std 0.0 \
    --env_action_bias_std 0.0 \
    --env_action_scale_range 1.0 1.0 \
    --env_action_delay_steps 0 0 \
    --chunk_size 0 \
    --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
    --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
    --x_range -0.8 0.8 \
    --signed_x \
    --x_abs_range 0.08 0.80 \
    --y_abs_range 0.05 0.30 \
    --yaw_abs_range 0.08 0.60 \
    --command_resample_interval_min 120 \
    --command_resample_interval_max 300 \
    --use_domain_randomization \
    --use_push_randomization \
    --use_observation_noise \
    --headless
done

MERGED_PATH="${RAW_DIR}/dataset.pt"
if [[ ! -f "${MERGED_PATH}" ]]; then
  # Loading and joining four 250k shards is CPU-heavy. Serialize this stage
  # across gaps so five GPU pipelines do not overload the shared host.
  run_stage merge_1m flock -x /tmp/go2_sim_gap_aligned_merge.lock "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_dataset/merge_go2_dataset_shards.py \
    --inputs "${SHARDS[@]}" \
    --output "${MERGED_PATH}" \
    --expected_shards 4 \
    --minimum_transitions 1000000
fi

if [[ ! -f "${SUBSET_PATH}" ]]; then
  run_stage slice_expert_25k "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_dataset/slice_go2_dataset_by_env_trajectories.py \
    --source_path "${MERGED_PATH}" \
    --save_path "${SUBSET_PATH}" \
    --num_trajectories 25 \
    --seed 20260714 \
    --selection random \
    --collector expert
fi

run_stage validate_expert_25k "${PYTHON_BIN}" \
  scripts/reinforcement_learning/rwm_dataset/validate_go2_expert_command_subset.py \
  --dataset_path "${SUBSET_PATH}" \
  --output_json "${RUN_ROOT}/dataset_quality.json" \
  --expected_transitions 25000

if [[ ! -f "${RWM_DIR}/stage/summary.json" ]]; then
  mkdir -p "${RWM_DIR}/stage"
  CURRENT_STAGE=train_rwm_baseline
  write_state "${CURRENT_STAGE}" running
  STAGE_DIR="${RWM_DIR}/stage" OUTPUT_DIR="${RWM_DIR}/runs" DATASET_PATH="${SUBSET_PATH}" \
    DEVICE=cuda:0 SEED=200 MAX_ITERATIONS=5000 BATCH_SIZE=1024 MICRO_BATCH_SIZE=256 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh \
    2>&1 | tee -a "${QUEUE_LOG}"
fi

MODEL_PATH="$(cat "${RWM_DIR}/stage/artifact_path.txt")"
if [[ ! -f "${POLICY_DIR}/stage/summary.json" ]]; then
  mkdir -p "${POLICY_DIR}/stage"
  CURRENT_STAGE=train_final_policy_rwm_p0
  write_state "${CURRENT_STAGE}" running
  STAGE_DIR="${POLICY_DIR}/stage" OUTPUT_DIR="${POLICY_DIR}/run" \
    DATASET_PATH="${SUBSET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 \
    DEVICE=cuda:0 SEED=300 NUM_ENV_STEPS=50000000 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
    2>&1 | tee -a "${QUEUE_LOG}"
fi

CURRENT_STAGE=done
write_state done completed
printf 'gap_id=%s\npayload_kg=%s\nrr_strength=%s\nraw_dataset=%s\nsubset_dataset=%s\nrwm_model=%s\nfinal_policy=%s\n' \
  "${GAP_ID}" "${PAYLOAD_KG}" "${RR_STRENGTH}" "${MERGED_PATH}" "${SUBSET_PATH}" \
  "${MODEL_PATH}" "$(cat "${POLICY_DIR}/stage/artifact_path.txt")" > "${RUN_ROOT}/summary.txt"
trap - EXIT
