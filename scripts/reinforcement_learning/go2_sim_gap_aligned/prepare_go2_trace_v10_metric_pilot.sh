#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${SIDE:?Set sim or real}"
: "${CONDITION:?Set condition}"
: "${RUN_ROOT:?Set a new metric-pilot root}"
: "${DATASET_PATH:?Set condition dataset}"
: "${MODEL_PATH:?Set frozen condition RWM}"
: "${INITIAL_POLICY_PATH:?Set common warmup}"
: "${V10_GPU_POOL:?Set physical GPU pool}"
PY="${PYTHON_BIN:-${REPO}/.venv/bin/python}"
PROTOCOL="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json"
METRIC_CONFIG="${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_metric_pilot.json"
SHARED="${RUN_ROOT}/shared/refresh_01"
mkdir -p "${SHARED}" "${RUN_ROOT}/metric/replay" "${RUN_ROOT}/random/replay"
for path in "${DATASET_PATH}" "${MODEL_PATH}" "${INITIAL_POLICY_PATH}/actor.pt" \
  "${INITIAL_POLICY_PATH}/replay_buffer.pt" "${PROTOCOL}" "${METRIC_CONFIG}"; do
  [[ -s "${path}" ]] || { echo "Missing metric-pilot input ${path}" >&2; exit 2; }
done
cd "${REPO}"

SIDE="${SIDE}" DATASET_PATH="${DATASET_PATH}" POLICY_PATH="${INITIAL_POLICY_PATH}" \
  CYCLE_ROOT="${SHARED}" REFRESH_CYCLE=1 V10_GPU_POOL="${V10_GPU_POOL}" REPO="${REPO}" \
  bash scripts/reinforcement_learning/go2_sim_gap_aligned/collect_go2_trace_v10_candidates.sh

for selection in metric random; do
  shard="${RUN_ROOT}/${selection}/replay/trace_shard.pt"
  summaries="${RUN_ROOT}/${selection}/replay/summaries.jsonl"
  if [[ ! -s "${shard}" ]]; then
    "${PY}" scripts/reinforcement_learning/rwm_trace/build_trace_replay_v10.py \
      --candidate_dataset "${SHARED}/candidates.pt" --target_dataset "${DATASET_PATH}" \
      --protocol "${PROTOCOL}" --metric_pilot_config "${METRIC_CONFIG}" \
      --refresh_cycle 1 --policy_checkpoint "${INITIAL_POLICY_PATH}" \
      --selection "${selection}" --selection_scope global --select_ratio 0.25 --seed 42 \
      --include_terminal_prefixes \
      --output "${shard}" --summary_jsonl "${summaries}"
  fi
  "${PY}" scripts/reinforcement_learning/rwm_trace/audit_v10_metric_pilot.py \
    --replay "${shard}" --summaries "${summaries}" --protocol "${PROTOCOL}" \
    --metric-config "${METRIC_CONFIG}" --selection "${selection}" \
    --output "${RUN_ROOT}/${selection}/replay/selection_audit.json"
  manifest="${RUN_ROOT}/${selection}/replay/staged_manifest.json"
  if [[ ! -s "${manifest}" ]]; then
    "${PY}" scripts/reinforcement_learning/rwm_trace/v10_replay_manifest.py stage \
      --output "${manifest}" --shard "${shard}" --cycle 1 --protocol "${PROTOCOL}" \
      --side "${SIDE}" --condition "${CONDITION}"
  fi
done

cat > "${RUN_ROOT}/replays_ready.env.new" <<EOF
passed=true
protocol_sha256=$(sha256sum "${PROTOCOL}" | awk '{print $1}')
metric_config_sha256=$(sha256sum "${METRIC_CONFIG}" | awk '{print $1}')
candidate=$(realpath "${SHARED}/candidates.pt")
metric_manifest=$(realpath "${RUN_ROOT}/metric/replay/staged_manifest.json")
random_manifest=$(realpath "${RUN_ROOT}/random/replay/staged_manifest.json")
EOF
mv "${RUN_ROOT}/replays_ready.env.new" "${RUN_ROOT}/replays_ready.env"
