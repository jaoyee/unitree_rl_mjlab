#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SIDE="${SIDE:?Set SIDE to sim or real}"
case "${SIDE}" in sim|real) ;; *) echo "SIDE must be sim or real" >&2; exit 2 ;; esac

DATASET="${DATASET_OVERRIDE:-${REPO}/logs/rwm_datasets_v4/${SIDE}_v4/pooled_40_20_20_10_10/dataset_pooled25k.pt}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${REPO}/logs/experiments/go2_pooled_v4/20260717_40_20_20_10_10/${SIDE}/rwm}"
STAGE_DIR="${RUN_ROOT}/stage"
OUTPUT_DIR="${RUN_ROOT}/runs"
PYTHON="${REPO}/.venv/bin/python"

[[ -s "${DATASET}" ]] || { echo "Missing pooled dataset ${DATASET}" >&2; exit 2; }
mkdir -p "${STAGE_DIR}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

if [[ ! -s "${STAGE_DIR}/summary.json" ]]; then
  STAGE_DIR="${STAGE_DIR}" OUTPUT_DIR="${OUTPUT_DIR}" DATASET_PATH="${DATASET}" \
    DEVICE=cuda:0 SEED=420 MAX_ITERATIONS=5000 BATCH_SIZE=1024 MICRO_BATCH_SIZE=256 \
    STATE_LOSS_IGNORED_INDICES="" SKIP_RWM_EVAL=1 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh
fi

MODEL="$(<"${STAGE_DIR}/artifact_path.txt")"
sha256sum "${DATASET}" "${MODEL}" > "${RUN_ROOT}/input_artifact_sha256.txt"
printf 'side=%s\ndataset=%s\nmodel=%s\nstatus=completed\n' \
  "${SIDE}" "${DATASET}" "${MODEL}" > "${RUN_ROOT}/rwm_summary.txt"
