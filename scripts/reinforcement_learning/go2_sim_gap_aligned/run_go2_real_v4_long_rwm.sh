#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
: "${GAP_ID:?Set GAP_ID to g0 or p5}"
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU}"

case "${GAP_ID}" in
  g0|p5) ;;
  *) echo "Unsupported GAP_ID=${GAP_ID}" >&2; exit 2 ;;
esac

cd "${REPO}"
DATASET="${DATASET_OVERRIDE:-${REPO}/logs/rwm_datasets_v4/real_v4/${GAP_ID}/dataset_partitioned25k.pt}"
RUN_ROOT="${REPO}/logs/experiments/go2_real_v4_t1_h100_r10/20260717_pilot/${GAP_ID}/rwm_long25k"
STAGE_DIR="${RUN_ROOT}/stage"
OUTPUT_DIR="${RUN_ROOT}/runs"
mkdir -p "${STAGE_DIR}"

if [[ ! -f "${DATASET}" ]]; then
  echo "Missing selected dataset: ${DATASET}" >&2
  exit 2
fi

export WANDB_MODE=offline
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

if [[ ! -s "${STAGE_DIR}/summary.json" ]]; then
  STAGE_DIR="${STAGE_DIR}" OUTPUT_DIR="${OUTPUT_DIR}" DATASET_PATH="${DATASET}" \
    DEVICE=cuda:0 SEED=420 MAX_ITERATIONS=5000 BATCH_SIZE=1024 MICRO_BATCH_SIZE=256 \
    bash scripts/reinforcement_learning/go2_normal_proprioceptive/03_train_normal_rwm.sh
fi

MODEL="$(cat "${STAGE_DIR}/artifact_path.txt")"
for horizon in 20 40 100 200; do
  output="${STAGE_DIR}/rwm_eval_h${horizon}.log"
  [[ -s "${output}" ]] && continue
  "${REPO}/.venv/bin/python" \
    scripts/reinforcement_learning/rwm_dataset/eval_world_model_go2_proprioceptive.py \
    --dataset_path "${DATASET}" --model_path "${MODEL}" \
    --num_eval_sequences 4096 --rollout_horizon "${horizon}" \
    --device cuda:0 --seed "$((421 + horizon))" > "${output}" 2>&1
done

printf 'dataset=%s\nmodel=%s\nstatus=completed\n' "${DATASET}" "${MODEL}" \
  > "${RUN_ROOT}/v4_rwm_summary.txt"
