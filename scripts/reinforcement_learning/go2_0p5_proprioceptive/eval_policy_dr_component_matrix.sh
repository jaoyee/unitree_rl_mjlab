#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a completed final-policy checkpoint directory.}"

RUN_ID="${RUN_ID:-policy_dr_matrix_$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-logs/evals/go2_0p5_policy_dr_matrix/${RUN_ID}}"
GPU_POOL="${GPU_POOL:-0 1 2 3}"
GPU_POOL="${GPU_POOL//,/ }"
EVAL_SEEDS="${EVAL_SEEDS:-123 456 789}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30m}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CONDITIONS="${CONDITIONS:-clean friction mass_com motor delay push observation full core_phys core_phys_obs}"
CONDITIONS="${CONDITIONS//,/ }"

read -r -a GPUS <<< "${GPU_POOL}"
read -r -a CONDITION_LIST <<< "${CONDITIONS}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_POOL must contain at least one GPU." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${EVAL_ROOT}"
GLOBAL_STATE="${EVAL_ROOT}/state.txt"
printf 'status=running\nstarted_at=%s\ncheckpoint=%s\n' \
  "$(date --iso-8601=seconds)" "${CHECKPOINT_PATH}" > "${GLOBAL_STATE}"

components_for_condition() {
  case "$1" in
    clean) echo "" ;;
    friction|mass_com|motor|delay|push|observation) echo "$1" ;;
    full) echo "all" ;;
    core_phys) echo "friction,mass_com,motor,push" ;;
    core_phys_obs) echo "friction,mass_com,motor,push,observation" ;;
    core_phys_half) echo "friction,mass_com,motor,push" ;;
    core_phys_half_obs) echo "friction,mass_com,motor,push,observation" ;;
    *) echo "Unknown evaluation condition: $1" >&2; return 2 ;;
  esac
}

scale_for_condition() {
  case "$1" in
    core_phys_half|core_phys_half_obs) echo "0.5" ;;
    *) echo "1.0" ;;
  esac
}

run_condition() {
  local condition="$1" gpu="$2" components randomization_scale seed output_json output_log
  components="$(components_for_condition "${condition}")"
  randomization_scale="$(scale_for_condition "${condition}")"
  printf 'condition=%s\ngpu=%s\ncomponents=%s\nrandomization_scale=%s\nstatus=running\n' \
    "${condition}" "${gpu}" "${components}" "${randomization_scale}" \
    > "${EVAL_ROOT}/${condition}.state"

  for seed in ${EVAL_SEEDS}; do
    output_json="${EVAL_ROOT}/${condition}__seed${seed}.json"
    output_log="${EVAL_ROOT}/${condition}__seed${seed}.log"
    command=(
      timeout "${EVAL_TIMEOUT}"
      "${PYTHON_BIN}"
      scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py
      --checkpoint_path "${CHECKPOINT_PATH}"
      --device cuda:0
      --num_envs "${EVAL_NUM_ENVS}"
      --steps "${EVAL_STEPS}"
      --seed "${seed}"
      --joint_strength_scales RR_calf_joint=0.5
      --output_json "${output_json}"
    )
    if [[ "${condition}" == "clean" ]]; then
      command+=(--clean)
    else
      command+=(
        --no-clean
        --randomization_preset friend_flat
        --randomization_components "${components}"
        --randomization_scale "${randomization_scale}"
      )
    fi

    echo "[$(date --iso-8601=seconds)] condition=${condition} seed=${seed} gpu=${gpu}" \
      | tee -a "${EVAL_ROOT}/matrix.log"
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      MUJOCO_GL=egl \
      OMP_NUM_THREADS=2 \
      MKL_NUM_THREADS=2 \
      NUMEXPR_NUM_THREADS=2 \
      "${command[@]}" > "${output_log}" 2>&1
  done

  printf 'condition=%s\ngpu=%s\ncomponents=%s\nrandomization_scale=%s\nstatus=completed\ncompleted_at=%s\n' \
    "${condition}" "${gpu}" "${components}" "${randomization_scale}" "$(date --iso-8601=seconds)" \
    > "${EVAL_ROOT}/${condition}.state"
}

declare -A JOB_PID=()
declare -A JOB_GPU=()
declare -A GPU_JOB=()
failed=0

reap_jobs() {
  local condition pid gpu exit_code
  for condition in "${!JOB_PID[@]}"; do
    pid="${JOB_PID[${condition}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        exit_code=0
      else
        exit_code=$?
        failed=1
        printf 'condition=%s\ngpu=%s\nstatus=failed\nexit_code=%s\n' \
          "${condition}" "${JOB_GPU[${condition}]}" "${exit_code}" \
          > "${EVAL_ROOT}/${condition}.state"
      fi
      gpu="${JOB_GPU[${condition}]}"
      echo "[$(date --iso-8601=seconds)] condition=${condition} exit_code=${exit_code}" \
        | tee -a "${EVAL_ROOT}/matrix.log"
      unset "GPU_JOB[${gpu}]"
      unset "JOB_PID[${condition}]"
      unset "JOB_GPU[${condition}]"
    fi
  done
}

acquire_gpu() {
  local gpu
  ACQUIRED_GPU=""
  while [[ -z "${ACQUIRED_GPU}" ]]; do
    reap_jobs
    for gpu in "${GPUS[@]}"; do
      if [[ -z "${GPU_JOB[${gpu}]:-}" ]]; then
        ACQUIRED_GPU="${gpu}"
        return 0
      fi
    done
    sleep 2
  done
}

for condition in "${CONDITION_LIST[@]}"; do
  if [[ -f "${EVAL_ROOT}/${condition}.state" ]] \
      && grep -q '^status=completed$' "${EVAL_ROOT}/${condition}.state"; then
    continue
  fi
  acquire_gpu
  gpu="${ACQUIRED_GPU}"
  run_condition "${condition}" "${gpu}" &
  JOB_PID[${condition}]=$!
  JOB_GPU[${condition}]="${gpu}"
  GPU_JOB[${gpu}]="${condition}"
done

while (( ${#JOB_PID[@]} > 0 )); do
  reap_jobs
  sleep 2
done

if (( failed != 0 )); then
  printf 'status=failed\ncompleted_at=%s\n' "$(date --iso-8601=seconds)" > "${GLOBAL_STATE}"
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_policy_dr_evaluations.py" \
  --eval_root "${EVAL_ROOT}" > "${EVAL_ROOT}/summary.log" 2>&1
printf 'status=completed\ncompleted_at=%s\n' "$(date --iso-8601=seconds)" > "${GLOBAL_STATE}"
echo "Policy DR evaluation matrix completed: ${EVAL_ROOT}"
