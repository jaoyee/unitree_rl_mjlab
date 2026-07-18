#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SIDE="${SIDE:?Set SIDE to sim or real}"
case "${SIDE}" in sim|real) ;; *) echo "SIDE must be sim or real" >&2; exit 2 ;; esac

DATASET="${DATASET_OVERRIDE:-${REPO}/logs/rwm_datasets_v4/${SIDE}_v4/pooled_40_20_20_10_10/dataset_pooled25k.pt}"
RWM_ROOT="${RWM_ROOT_OVERRIDE:-${REPO}/logs/experiments/go2_pooled_v4/20260717_40_20_20_10_10/${SIDE}/rwm/runs}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${REPO}/logs/experiments/go2_pooled_v4/20260717_40_20_20_10_10/${SIDE}/policies/rwm_baseline}"

mapfile -t MODELS < <(find "${RWM_ROOT}" -mindepth 2 -maxdepth 2 -type f -name model_5000.pt -print)
if [[ "${#MODELS[@]}" -ne 1 ]]; then
  echo "Expected exactly one model_5000.pt under ${RWM_ROOT}, found ${#MODELS[@]}" >&2
  exit 2
fi
MODEL="${MODELS[0]}"

STAGE_DIR="${RUN_ROOT}/stage" OUTPUT_DIR="${RUN_ROOT}/run" \
DATASET_PATH="${DATASET}" MODEL_PATH="${MODEL}" P_ID=P0 DEVICE=cuda:0 \
SEED=300 NUM_ENV_STEPS=50000000 \
LIN_VEL_X_RANGE="-0.5 0.5" LIN_VEL_Y_RANGE="-0.2 0.2" ANG_VEL_Z_RANGE="-0.4 0.4" \
bash "${REPO}/scripts/reinforcement_learning/go2_normal_proprioceptive/04_train_normal_policy.sh"
