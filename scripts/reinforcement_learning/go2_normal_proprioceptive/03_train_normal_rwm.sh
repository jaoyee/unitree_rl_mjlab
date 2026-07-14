#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${STAGE_DIR:?Set STAGE_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${DATASET_PATH:?Set DATASET_PATH}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SEED="${SEED:-200}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-256}"
NUM_EVAL_SEQUENCES="${NUM_EVAL_SEQUENCES:-4096}"
STATE_LOSS_IGNORED_INDICES="${STATE_LOSS_IGNORED_INDICES:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi
OUTPUT_RELATIVE="${OUTPUT_DIR#${REPO_ROOT}/}"
if [[ "${OUTPUT_RELATIVE}" == *0p5* || "${OUTPUT_RELATIVE}" == *broken* ]]; then
  echo "Normal RWM output path contains a forbidden 0.5/broken marker: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "RWM output directory is not empty: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${STAGE_DIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2_proprioceptive.py \
  --config_path scripts/reinforcement_learning/rwm_dataset/configs/go2_offline_world_model_proprioceptive.yaml \
  --dataset_path "${DATASET_PATH}" \
  --save_dir "${OUTPUT_DIR}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --batch_size "${BATCH_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --device "${DEVICE}" \
  --save_interval 500 \
  --log_interval 50 \
  --overrides "seed=${SEED}" \
  --overrides "masked_joint_names=[]" \
  --overrides "action_mask_indices=[]" \
  --overrides "system_dynamics.state_loss_ignored_indices=[${STATE_LOSS_IGNORED_INDICES}]"

mapfile -t RUN_DIRS < <(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -print)
if [[ "${#RUN_DIRS[@]}" -ne 1 ]]; then
  echo "Expected exactly one RWM run directory, found ${#RUN_DIRS[@]} in ${OUTPUT_DIR}" >&2
  exit 1
fi
RUN_DIR="${RUN_DIRS[0]}"
MODEL_PATH="${RUN_DIR}/model_${MAX_ITERATIONS}.pt"
if [[ ! -f "${MODEL_PATH}" || ! -f "${RUN_DIR}/latest.pt" ]]; then
  echo "Incomplete RWM artifact: ${MODEL_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/reinforcement_learning/rwm_dataset/eval_world_model_go2_proprioceptive.py \
  --dataset_path "${DATASET_PATH}" \
  --model_path "${MODEL_PATH}" \
  --num_eval_sequences "${NUM_EVAL_SEQUENCES}" \
  --rollout_horizon 20 \
  --device "${DEVICE}" \
  --seed "$((SEED + 1))" \
  > "${STAGE_DIR}/rwm_eval.log" 2>&1

MODEL_PATH="$(realpath "${MODEL_PATH}")"
MODEL_SHA256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
DATASET_SHA256="$(sha256sum "${DATASET_PATH}" | awk '{print $1}')"
printf '%s\n' "${MODEL_PATH}" > "${STAGE_DIR}/artifact_path.txt"
printf '%s\n' "${MODEL_SHA256}" > "${STAGE_DIR}/artifact_sha256.txt"

"${PYTHON_BIN}" - "${STAGE_DIR}/summary.json" "${MODEL_PATH}" "${MODEL_SHA256}" \
  "$(realpath "${DATASET_PATH}")" "${DATASET_SHA256}" "${SEED}" "${MAX_ITERATIONS}" \
  "${STATE_LOSS_IGNORED_INDICES}" <<'PY'
import json
import sys
from pathlib import Path

output, artifact, digest, dataset, dataset_digest, seed, iterations, ignored_indices = sys.argv[1:]
ignored = [int(value) for value in ignored_indices.replace(",", " ").split() if value]
Path(output).write_text(json.dumps({
    "status": "completed",
    "artifact_path": artifact,
    "artifact_sha256": digest,
    "dataset_path": dataset,
    "dataset_sha256": dataset_digest,
    "seed": int(seed),
    "max_iterations": int(iterations),
    "state_loss_ignored_indices": ignored,
    "masked_joint_names": [],
    "action_mask_indices": [],
    "evaluation_log": str(Path(output).parent / "rwm_eval.log"),
}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "rwm_artifact=${MODEL_PATH}"
echo "rwm_sha256=${MODEL_SHA256}"
