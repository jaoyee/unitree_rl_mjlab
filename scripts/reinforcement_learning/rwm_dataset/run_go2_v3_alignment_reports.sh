#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SIM_DATASET_ROOT="${SIM_DATASET_ROOT:-${REPO_ROOT}/logs/rwm_datasets_v2/sim_v2}"
REAL_DATASET_ROOT="${REAL_DATASET_ROOT:-${REPO_ROOT}/logs/transfer_real_trace_inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/experiments/go2_v3_t2_long_failure/dynamics_alignment}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

mkdir -p "${OUTPUT_ROOT}"
for condition in g0 rr03 rr05 p5 p75; do
  sim="${SIM_DATASET_ROOT}/${condition}/dataset.pt"
  real="${REAL_DATASET_ROOT}/${condition}/dataset.pt"
  [[ -f "${sim}" ]] || { echo "Missing sim dataset: ${sim}" >&2; exit 2; }
  [[ -f "${real}" ]] || { echo "Missing real dataset: ${real}" >&2; exit 2; }
  "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_dataset/compare_go2_sim_real_dynamics.py \
    --sim-dataset "${sim}" \
    --real-dataset "${real}" \
    --history-length 40 \
    --output "${OUTPUT_ROOT}/${condition}.json"
done
date --iso-8601=seconds > "${OUTPUT_ROOT}/COMPLETE"
echo "${OUTPUT_ROOT}"
