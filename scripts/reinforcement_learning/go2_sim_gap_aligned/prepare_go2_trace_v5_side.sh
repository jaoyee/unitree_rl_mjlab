#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set SIDE=sim or SIDE=real}"
: "${DATASET_PATH:?Set the immutable 25K dataset path}"
: "${RUN_ROOT:?Set a new V5 side root}"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU}"
case "${SIDE}" in sim|real) ;; *) exit 2 ;; esac

export WANDB_MODE=offline
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd "${REPO}"
[[ -s "${DATASET_PATH}" ]] || { echo "Missing dataset ${DATASET_PATH}" >&2; exit 2; }
mkdir -p "${RUN_ROOT}/rwm/stage" "${RUN_ROOT}/common_warmup/stage" \
  "${RUN_ROOT}/baseline/stage"

if [[ ! -s "${RUN_ROOT}/rwm/stage/summary.json" ]]; then
  STAGE_DIR="${RUN_ROOT}/rwm/stage" OUTPUT_DIR="${RUN_ROOT}/rwm/runs" \
    DATASET_PATH="${DATASET_PATH}" DEVICE=cuda:0 SEED=420 MAX_ITERATIONS=5000 \
    BATCH_SIZE=1024 MICRO_BATCH_SIZE=256 STATE_LOSS_IGNORED_INDICES='' SKIP_RWM_EVAL=1 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh \
    2>&1 | tee "${RUN_ROOT}/rwm.log"
fi
MODEL_PATH="$(<"${RUN_ROOT}/rwm/stage/artifact_path.txt")"
[[ -s "${MODEL_PATH}" ]] || { echo "RWM artifact missing: ${MODEL_PATH}" >&2; exit 2; }

if [[ ! -s "${RUN_ROOT}/common_warmup/stage/summary.json" ]]; then
  STAGE_DIR="${RUN_ROOT}/common_warmup/stage" OUTPUT_DIR="${RUN_ROOT}/common_warmup/run" \
    DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 DEVICE=cuda:0 \
    SEED=300 NUM_ENV_STEPS=10000000 SAVE_REPLAY_BUFFER=true \
    LIN_VEL_X_RANGE='-0.5 0.5' LIN_VEL_Y_RANGE='-0.2 0.2' ANG_VEL_Z_RANGE='-0.4 0.4' \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
    2>&1 | tee "${RUN_ROOT}/common_warmup.log"
fi
INITIAL_POLICY_PATH="$(<"${RUN_ROOT}/common_warmup/stage/artifact_path.txt")"
for file in actor.pt agent_state.pt reward_normalizer.pt replay_buffer.pt; do
  [[ -s "${INITIAL_POLICY_PATH}/${file}" ]] || { echo "Warm-up checkpoint misses ${file}" >&2; exit 2; }
done

{
  printf 'DATASET_PATH=%q\n' "$(realpath "${DATASET_PATH}")"
  printf 'MODEL_PATH=%q\n' "$(realpath "${MODEL_PATH}")"
  printf 'INITIAL_POLICY_PATH=%q\n' "$(realpath "${INITIAL_POLICY_PATH}")"
} > "${RUN_ROOT}/ready.env.tmp"
mv "${RUN_ROOT}/ready.env.tmp" "${RUN_ROOT}/ready.env"

# The baseline receives the same dataset, model, seed, command range and total
# 50M policy-environment steps as each TRACE branch, but no TRACE replay.
if [[ ! -s "${RUN_ROOT}/baseline/stage/summary.json" ]]; then
  STAGE_DIR="${RUN_ROOT}/baseline/stage" OUTPUT_DIR="${RUN_ROOT}/baseline/run" \
    DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" P_ID=P0 DEVICE=cuda:0 \
    SEED=300 NUM_ENV_STEPS=50000000 SAVE_REPLAY_BUFFER=false \
    LIN_VEL_X_RANGE='-0.5 0.5' LIN_VEL_Y_RANGE='-0.2 0.2' ANG_VEL_Z_RANGE='-0.4 0.4' \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh \
    2>&1 | tee "${RUN_ROOT}/baseline.log"
fi
printf 'side=%s\nstatus=completed\n' "${SIDE}" > "${RUN_ROOT}/prepare_completed.txt"
