#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/unitree_rl_mjlab_normal_fixstand_v2
RUN_ROOT="${REPO_ROOT}/logs/experiments/go2_real_trace_rwm/20260714_g0_25k_v1"
DATASET_PATH="${REPO_ROOT}/logs/rwm_datasets/go2_real_g0_baseline_25k_20260714/dataset.pt"
STAGE_DIR="${RUN_ROOT}/rwm_baseline/stage"
OUTPUT_DIR="${RUN_ROOT}/rwm_baseline/runs"
STATE_FILE="${RUN_ROOT}/rwm_state.txt"
EXIT_FILE="${RUN_ROOT}/rwm_exit_code.txt"
LOG_FILE="${RUN_ROOT}/rwm.log"

mkdir -p "${RUN_ROOT}" "${STAGE_DIR}"
cd "${REPO_ROOT}"
rm -f "${EXIT_FILE}"
printf 'time=%s\nstage=train_real_rwm\nstatus=running\ngpu=1\n' \
  "$(date --iso-8601=seconds)" > "${STATE_FILE}"

set +e
CUDA_VISIBLE_DEVICES=1 \
STAGE_DIR="${STAGE_DIR}" OUTPUT_DIR="${OUTPUT_DIR}" DATASET_PATH="${DATASET_PATH}" \
DEVICE=cuda:0 SEED=200 MAX_ITERATIONS=5000 BATCH_SIZE=1024 MICRO_BATCH_SIZE=256 \
STATE_LOSS_IGNORED_INDICES="" \
bash scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh \
  2>&1 | tee -a "${LOG_FILE}"
code=${PIPESTATUS[0]}
set -e

printf '%s\n' "${code}" > "${EXIT_FILE}"
if (( code == 0 )); then
  printf 'time=%s\nstage=train_real_rwm\nstatus=completed\ngpu=1\n' \
    "$(date --iso-8601=seconds)" > "${STATE_FILE}"
else
  printf 'time=%s\nstage=train_real_rwm\nstatus=failed\ngpu=1\nexit_code=%s\n' \
    "$(date --iso-8601=seconds)" "${code}" > "${STATE_FILE}"
fi
exit "${code}"
