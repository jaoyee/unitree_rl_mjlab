#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_GROUP="${RUN_GROUP:-friend_flat_dr_strict2x2_20260710_1630}"
EVAL_ID="${EVAL_ID:-real_mjlab_matrix_$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-logs/evals/go2_0p5_friend_flat_dr/${RUN_GROUP}/${EVAL_ID}}"
GPU_POOL="${GPU_POOL:-2 4 6 7}"
GPU_POOL="${GPU_POOL//,/ }"
EVAL_SEEDS="${EVAL_SEEDS:-123 456 789}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-30m}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
RANDOMIZATION_COMPONENTS="${RANDOMIZATION_COMPONENTS:-all}"
RANDOMIZATION_SCALE="${RANDOMIZATION_SCALE:-1.0}"

POLICIES=(
  base_final_clean
  base_final_interface
  action_noise_final_clean
  action_noise_final_interface
)

read -r -a GPUS <<< "${GPU_POOL}"
if (( ${#GPUS[@]} < ${#POLICIES[@]} )); then
  echo "GPU_POOL must contain at least ${#POLICIES[@]} GPU ids." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

ARTIFACT_ROOT="logs/experiments/go2_0p5_friend_flat_dr/${RUN_GROUP}/policies"
mkdir -p "${EVAL_ROOT}"
printf 'status=running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${EVAL_ROOT}/state.txt"

resolve_checkpoint() {
  local policy="$1" checkpoint
  checkpoint="$(find "${ARTIFACT_ROOT}/${policy}" -mindepth 2 -maxdepth 2 -type d -name step48828 | sort | tail -n 1)"
  if [[ -z "${checkpoint}" ]]; then
    echo "Missing step48828 checkpoint for ${policy}" >&2
    return 1
  fi
  printf '%s\n' "${checkpoint}"
}

run_policy() {
  local policy="$1" gpu="$2" checkpoint condition seed output_json output_log
  checkpoint="$(resolve_checkpoint "${policy}")"
  printf 'policy=%s\ngpu=%s\ncheckpoint=%s\nstatus=running\n' \
    "${policy}" "${gpu}" "${checkpoint}" > "${EVAL_ROOT}/${policy}.state"

  for condition in clean friend_flat; do
    for seed in ${EVAL_SEEDS}; do
      output_json="${EVAL_ROOT}/${policy}__${condition}__seed${seed}.json"
      output_log="${EVAL_ROOT}/${policy}__${condition}__seed${seed}.log"
      command=(
        timeout "${EVAL_TIMEOUT}"
        "${PYTHON_BIN}"
        scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py
        --checkpoint_path "${checkpoint}"
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
          --randomization_components "${RANDOMIZATION_COMPONENTS}"
          --randomization_scale "${RANDOMIZATION_SCALE}"
        )
      fi

      echo "[$(date --iso-8601=seconds)] policy=${policy} condition=${condition} seed=${seed} gpu=${gpu}" \
        | tee -a "${EVAL_ROOT}/matrix.log"
      if ! env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        MUJOCO_GL=egl \
        OMP_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 \
        NUMEXPR_NUM_THREADS=2 \
        "${command[@]}" > "${output_log}" 2>&1; then
        printf 'policy=%s\ngpu=%s\ncheckpoint=%s\nstatus=failed\ncondition=%s\nseed=%s\nlog=%s\n' \
          "${policy}" "${gpu}" "${checkpoint}" "${condition}" "${seed}" "${output_log}" \
          > "${EVAL_ROOT}/${policy}.state"
        return 1
      fi
    done
  done

  printf 'policy=%s\ngpu=%s\ncheckpoint=%s\nstatus=completed\ncompleted_at=%s\n' \
    "${policy}" "${gpu}" "${checkpoint}" "$(date --iso-8601=seconds)" \
    > "${EVAL_ROOT}/${policy}.state"
}

pids=()
for index in "${!POLICIES[@]}"; do
  run_policy "${POLICIES[${index}]}" "${GPUS[${index}]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  printf 'status=failed\ncompleted_at=%s\n' "$(date --iso-8601=seconds)" > "${EVAL_ROOT}/state.txt"
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_policy_dr_evaluations.py" \
  --eval_root "${EVAL_ROOT}" > "${EVAL_ROOT}/summary.log" 2>&1
printf 'status=completed\ncompleted_at=%s\n' "$(date --iso-8601=seconds)" > "${EVAL_ROOT}/state.txt"
echo "All real-MJLab evaluations completed: ${EVAL_ROOT}"
