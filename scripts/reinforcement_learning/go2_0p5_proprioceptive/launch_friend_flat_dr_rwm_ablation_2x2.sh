#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${EXPERT_POLICY_PATH:?Set EXPERT_POLICY_PATH to the best 0.5-strength expert checkpoint directory.}"

RUN_GROUP="${RUN_GROUP:-friend_flat_dr_2x2_$(date +%Y%m%d_%H%M%S)}"
GPU_POOL="${GPU_POOL:-0 1 2 3 4 5 6 7}"
# Comma-separated input survives SSH/Docker environment forwarding without
# fragile nested quoting; internally the scheduler still consumes a word list.
GPU_POOL="${GPU_POOL//,/ }"
GPU_MAX_MEMORY_USED_MIB="${GPU_MAX_MEMORY_USED_MIB:-2500}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-15}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-true}"
CONFIRM_RUN="${CONFIRM_RUN:-false}"
AUTO_TMUX="${AUTO_TMUX:-true}"
WATCHER_CHILD="${WATCHER_CHILD:-false}"
WATCHER_SESSION="${WATCHER_SESSION:-go2_dr_${RUN_GROUP}}"
WATCHER_SESSION="${WATCHER_SESSION//[^a-zA-Z0-9_-]/_}"
WATCHER_SESSION="${WATCHER_SESSION:0:80}"

COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-1024}"
COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS:-1000000}"
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-200000}"
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-5000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE:-256}"
SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS:-1024}"
SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS:-50000000}"
REUSE_EXISTING_DATASETS="${REUSE_EXISTING_DATASETS:-true}"
COLLECT_RANDOMIZATION_COMPONENTS="${COLLECT_RANDOMIZATION_COMPONENTS:-all}"
COLLECT_RANDOMIZATION_SCALE="${COLLECT_RANDOMIZATION_SCALE:-1.0}"

# A real run must keep the dependency watcher alive after the SSH client or
# local machine disconnects. Dry-runs stay in the foreground for inspection.
if [[ "${DRY_RUN}" != "true" && "${CONFIRM_RUN}" == "true" && "${AUTO_TMUX}" == "true" \
      && "${WATCHER_CHILD}" != "true" && -z "${TMUX:-}" ]]; then
  if tmux has-session -t "${WATCHER_SESSION}" 2>/dev/null; then
    echo "watcher session already exists: ${WATCHER_SESSION}"
    echo "attach: tmux attach -t ${WATCHER_SESSION}"
    exit 0
  fi
  printf -v child_command \
    'cd %q && env EXPERT_POLICY_PATH=%q RUN_GROUP=%q GPU_POOL=%q GPU_MAX_MEMORY_USED_MIB=%q GPU_MAX_UTILIZATION=%q GPU_POLL_SECONDS=%q COLLECT_NUM_ENVS=%q COLLECT_NUM_TRANSITIONS=%q COLLECT_CHUNK_SIZE=%q WM_MAX_ITERATIONS=%q WM_BATCH_SIZE=%q WM_MICRO_BATCH_SIZE=%q SAC_NUM_IMAGINATION_ENVS=%q SAC_NUM_ENV_STEPS=%q REUSE_EXISTING_DATASETS=%q COLLECT_RANDOMIZATION_COMPONENTS=%q COLLECT_RANDOMIZATION_SCALE=%q DRY_RUN=false CONFIRM_RUN=true AUTO_TMUX=false WATCHER_CHILD=true WATCHER_SESSION=%q bash %q' \
    "${REPO_ROOT}" "${EXPERT_POLICY_PATH}" "${RUN_GROUP}" "${GPU_POOL}" \
    "${GPU_MAX_MEMORY_USED_MIB}" "${GPU_MAX_UTILIZATION}" "${GPU_POLL_SECONDS}" \
    "${COLLECT_NUM_ENVS}" "${COLLECT_NUM_TRANSITIONS}" "${COLLECT_CHUNK_SIZE}" \
    "${WM_MAX_ITERATIONS}" "${WM_BATCH_SIZE}" "${WM_MICRO_BATCH_SIZE}" \
    "${SAC_NUM_IMAGINATION_ENVS}" "${SAC_NUM_ENV_STEPS}" "${REUSE_EXISTING_DATASETS}" \
    "${COLLECT_RANDOMIZATION_COMPONENTS}" \
    "${COLLECT_RANDOMIZATION_SCALE}" \
    "${WATCHER_SESSION}" "${BASH_SOURCE[0]}"
  tmux new-session -d -s "${WATCHER_SESSION}" "${child_command}"
  echo "started watcher session: ${WATCHER_SESSION}"
  echo "attach: tmux attach -t ${WATCHER_SESSION}"
  exit 0
fi

LAUNCH_ROOT="logs/queues/go2_0p5_friend_flat_dr/${RUN_GROUP}"
ARTIFACT_ROOT="logs/experiments/go2_0p5_friend_flat_dr/${RUN_GROUP}"
SUMMARY_FILE="${LAUNCH_ROOT}/launch_summary.txt"
mkdir -p "${LAUNCH_ROOT}/jobs" "${ARTIFACT_ROOT}"

BASE_DATASET="${ARTIFACT_ROOT}/collect_base/dataset.pt"
NOISE_DATASET="${ARTIFACT_ROOT}/collect_action_noise/dataset.pt"
BASE_WM="${ARTIFACT_ROOT}/collect_base/world_model"
NOISE_WM="${ARTIFACT_ROOT}/collect_action_noise/world_model"

BASE_SKIP_COLLECT=false
NOISE_SKIP_COLLECT=false
if [[ "${REUSE_EXISTING_DATASETS}" == "true" && -f "${BASE_DATASET}" ]]; then
  BASE_SKIP_COLLECT=true
fi
if [[ "${REUSE_EXISTING_DATASETS}" == "true" && -f "${NOISE_DATASET}" ]]; then
  NOISE_SKIP_COLLECT=true
fi

declare -A RESERVED_GPU=()
declare -A JOB_PID=()
declare -A JOB_GPU=()
declare -A JOB_QUEUE=()
ACQUIRED_GPU=""

{
  echo "run_group=${RUN_GROUP}"
  echo "repo_root=${REPO_ROOT}"
  echo "expert_policy_path=${EXPERT_POLICY_PATH}"
  echo "gpu_pool=${GPU_POOL}"
  echo "gpu_max_memory_used_mib=${GPU_MAX_MEMORY_USED_MIB}"
  echo "gpu_max_utilization=${GPU_MAX_UTILIZATION}"
  echo "dry_run=${DRY_RUN}"
  echo "confirm_run=${CONFIRM_RUN}"
  echo "reuse_existing_datasets=${REUSE_EXISTING_DATASETS}"
  echo "collect_randomization_components=${COLLECT_RANDOMIZATION_COMPONENTS}"
  echo "collect_randomization_scale=${COLLECT_RANDOMIZATION_SCALE}"
  echo "base_skip_collect=${BASE_SKIP_COLLECT}"
  echo "noise_skip_collect=${NOISE_SKIP_COLLECT}"
  echo "watcher_session=${WATCHER_SESSION}"
  echo "base_dataset=${BASE_DATASET}"
  echo "base_world_model=${BASE_WM}"
  echo "noise_dataset=${NOISE_DATASET}"
  echo "noise_world_model=${NOISE_WM}"
  echo "design=two shared collection/RWM branches, each split into clean/interface policies"
  echo "started_at=$(date --iso-8601=seconds)"
} > "${SUMMARY_FILE}"

gpu_is_idle() {
  local gpu="$1" values memory_used utilization
  values="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" || return 1
  IFS=, read -r memory_used utilization <<< "${values}"
  memory_used="${memory_used//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  [[ "${memory_used}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || return 1
  (( memory_used <= GPU_MAX_MEMORY_USED_MIB && utilization <= GPU_MAX_UTILIZATION ))
}

reap_finished_jobs() {
  local job pid gpu
  for job in "${!JOB_PID[@]}"; do
    pid="${JOB_PID[${job}]}"
    [[ "${pid}" == "0" ]] && continue
    if ! kill -0 "${pid}" 2>/dev/null; then
      if ! wait "${pid}"; then
        echo "job=${job} status=failed log=${JOB_QUEUE[${job}]}/launcher.log" | tee -a "${SUMMARY_FILE}"
        return 1
      fi
      JOB_PID["${job}"]=0
      gpu="${JOB_GPU[${job}]}"
      unset 'RESERVED_GPU['"${gpu}"']'
      echo "job=${job} process_finished_at=$(date --iso-8601=seconds)" | tee -a "${SUMMARY_FILE}"
    fi
  done
}

acquire_gpu() {
  local gpu
  while true; do
    reap_finished_jobs
    for gpu in ${GPU_POOL}; do
      [[ -n "${RESERVED_GPU[${gpu}]:-}" ]] && continue
      if [[ "${DRY_RUN}" == "true" ]] || gpu_is_idle "${gpu}"; then
        RESERVED_GPU["${gpu}"]=1
        ACQUIRED_GPU="${gpu}"
        return 0
      fi
    done
    echo "[$(date --iso-8601=seconds)] waiting for an idle GPU" | tee -a "${SUMMARY_FILE}"
    sleep "${GPU_POLL_SECONDS}"
  done
}

launch_job() {
  local job="$1" variant="$2" dataset_path="$3" wm_save_base="$4" sac_save_path="$5"
  local skip_collect="$6" skip_wm="$7" skip_policy="$8"
  local queue_root command_file gpu
  acquire_gpu
  gpu="${ACQUIRED_GPU}"
  queue_root="${LAUNCH_ROOT}/jobs/${job}"
  command_file="${queue_root}/command.sh"
  mkdir -p "${queue_root}"
  cat > "${command_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
env \
  EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH}" \
  VARIANT="${variant}" \
  RUN_ID="${RUN_GROUP}_${job}" \
  QUEUE_ROOT="${queue_root}/pipeline" \
  DATASET_PATH="${dataset_path}" \
  WM_SAVE_BASE="${wm_save_base}" \
  SAC_SAVE_PATH="${sac_save_path}" \
  SKIP_COLLECT="${skip_collect}" \
  SKIP_WM="${skip_wm}" \
  SKIP_POLICY="${skip_policy}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS}" \
  COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS}" \
  COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE}" \
  WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS}" \
  WM_BATCH_SIZE="${WM_BATCH_SIZE}" \
  WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE}" \
  SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS}" \
  SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS}" \
  COLLECT_RANDOMIZATION_COMPONENTS="${COLLECT_RANDOMIZATION_COMPONENTS}" \
  COLLECT_RANDOMIZATION_SCALE="${COLLECT_RANDOMIZATION_SCALE}" \
  DRY_RUN="${DRY_RUN}" \
  CONFIRM_RUN="${CONFIRM_RUN}" \
  bash scripts/reinforcement_learning/go2_0p5_proprioceptive/run_friend_flat_dr_rwm_ablation.sh
EOF
  chmod +x "${command_file}"
  JOB_GPU["${job}"]="${gpu}"
  JOB_QUEUE["${job}"]="${queue_root}"
  echo "job=${job} variant=${variant} gpu=${gpu} command=${command_file}" | tee -a "${SUMMARY_FILE}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    bash "${command_file}" > "${queue_root}/launcher.log" 2>&1
    JOB_PID["${job}"]=0
    unset 'RESERVED_GPU['"${gpu}"']'
  else
    bash "${command_file}" > "${queue_root}/launcher.log" 2>&1 &
    JOB_PID["${job}"]=$!
  fi
}

wait_for_jobs() {
  local job pid gpu status
  for job in "$@"; do
    pid="${JOB_PID[${job}]}"
    gpu="${JOB_GPU[${job}]}"
    if [[ "${pid}" != "0" ]]; then
      if ! wait "${pid}"; then
        echo "job=${job} status=failed log=${JOB_QUEUE[${job}]}/launcher.log" | tee -a "${SUMMARY_FILE}"
        return 1
      fi
    fi
    status="$(grep "^status=" "${JOB_QUEUE[${job}]}/pipeline/state.txt" | tail -n 1 | cut -d= -f2)"
    if [[ "${status}" != "completed" && "${status}" != "dry_run" ]]; then
      echo "job=${job} unexpected_status=${status}" | tee -a "${SUMMARY_FILE}"
      return 1
    fi
    unset 'RESERVED_GPU['"${gpu}"']'
    echo "job=${job} status=${status} completed_at=$(date --iso-8601=seconds)" | tee -a "${SUMMARY_FILE}"
  done
}

launch_job upstream_base collect_dr_final_clean "${BASE_DATASET}" "${BASE_WM}" unused "${BASE_SKIP_COLLECT}" false true
launch_job upstream_action_noise collect_dr_action_noise_final_clean "${NOISE_DATASET}" "${NOISE_WM}" unused "${NOISE_SKIP_COLLECT}" false true
wait_for_jobs upstream_base upstream_action_noise

launch_job base_final_clean collect_dr_final_clean "${BASE_DATASET}" "${BASE_WM}" "${ARTIFACT_ROOT}/policies/base_final_clean/TIMESTAMP" true true false
launch_job base_final_interface collect_dr_final_interface "${BASE_DATASET}" "${BASE_WM}" "${ARTIFACT_ROOT}/policies/base_final_interface/TIMESTAMP" true true false
launch_job action_noise_final_clean collect_dr_action_noise_final_clean "${NOISE_DATASET}" "${NOISE_WM}" "${ARTIFACT_ROOT}/policies/action_noise_final_clean/TIMESTAMP" true true false
launch_job action_noise_final_interface collect_dr_action_noise_final_interface "${NOISE_DATASET}" "${NOISE_WM}" "${ARTIFACT_ROOT}/policies/action_noise_final_interface/TIMESTAMP" true true false
wait_for_jobs base_final_clean base_final_interface action_noise_final_clean action_noise_final_interface

echo "completed_at=$(date --iso-8601=seconds)" | tee -a "${SUMMARY_FILE}"
echo "All strict 2x2 jobs completed."
