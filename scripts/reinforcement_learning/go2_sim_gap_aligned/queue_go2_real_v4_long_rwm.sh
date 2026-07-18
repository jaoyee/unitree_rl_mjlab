#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2}"
: "${GAP_ID:?Set GAP_ID}"
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU}"
cd "${REPO}"

dataset="${DATASET_OVERRIDE:-logs/rwm_datasets_v4/real_v4/${GAP_ID}/dataset_partitioned25k.pt}"
selection_log="logs/experiments/go2_real_v4_t1_h100_r10/20260717_pilot/selection/${GAP_ID}_partitioned25k.log"
for _ in $(seq 1 240); do
  [[ -s "${dataset}" ]] && break
  if grep -q Traceback "${selection_log}" 2>/dev/null; then
    echo "Selection failed for ${GAP_ID}; see ${selection_log}" >&2
    exit 3
  fi
  sleep 30
done
[[ -s "${dataset}" ]] || { echo "Timed out waiting for ${dataset}" >&2; exit 4; }

audit="logs/experiments/go2_real_v4_t1_h100_r10/20260717_pilot/audits/${GAP_ID}_long25k.json"
"${REPO}/.venv/bin/python" \
  scripts/reinforcement_learning/rwm_dataset/audit_go2_sequence_quality_v4.py \
  --input "${dataset}" --output-json "${audit}"

REPO="${REPO}" GAP_ID="${GAP_ID}" DATASET_OVERRIDE="${dataset}" \
  bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_real_v4_long_rwm.sh
