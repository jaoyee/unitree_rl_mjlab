#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${V10_ARTIFACT_ENV:?Set server-local artifact manifest}"
: "${V10_RUN_BASE:?Set a new metric-pilot run base}"
: "${V10_GPU_POOL:?Set physical GPU pool}"
SIDE="${SIDE:-real}"
CONDITION="${CONDITION:-rr05}"
source "${V10_ARTIFACT_ENV}"
prefix="${SIDE^^}_${CONDITION^^}"
dataset_var="${prefix}_DATASET"; model_var="${prefix}_MODEL"; warmup_var="${prefix}_WARMUP"
DATASET_PATH="${!dataset_var:-}"; MODEL_PATH="${!model_var:-}"; INITIAL_POLICY_PATH="${!warmup_var:-}"
for path in "${DATASET_PATH}" "${MODEL_PATH}" "${INITIAL_POLICY_PATH}/actor.pt"; do
  [[ -s "${path}" ]] || { echo "Missing metric-pilot artifact ${path}" >&2; exit 2; }
done
mkdir -p "${V10_RUN_BASE}"
controller="v10m_${SIDE}_${CONDITION}_controller"
for session in "${controller}" "v10m_${SIDE}_${CONDITION}_metric" \
  "v10m_${SIDE}_${CONDITION}_random" "v10m_${SIDE}_${CONDITION}_control"; do
  tmux has-session -t "${session}" 2>/dev/null && { echo "Session exists: ${session}" >&2; exit 2; }
done

tmux new-session -d -s "${controller}" env \
  REPO="${REPO}" SIDE="${SIDE}" CONDITION="${CONDITION}" RUN_ROOT="${V10_RUN_BASE}" \
  DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
  INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" V10_GPU_POOL="${V10_GPU_POOL}" \
  bash -lc '
set -euo pipefail
control_session="v10m_${SIDE}_${CONDITION}_control"
tmux new-session -d -s "${control_session}" env \
  REPO="${REPO}" SIDE="${SIDE}" CONDITION="${CONDITION}" RUN_ROOT="${RUN_ROOT}/control" \
  DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
  INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" V10_GPU_POOL="${V10_GPU_POOL}" RUN_KIND=pilot \
  V10_EVAL_SEEDS="901 902 903" \
  bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_no_trace_control.sh"

REPO="${REPO}" SIDE="${SIDE}" CONDITION="${CONDITION}" RUN_ROOT="${RUN_ROOT}" \
  DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
  INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" V10_GPU_POOL="${V10_GPU_POOL}" \
  bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/prepare_go2_trace_v10_metric_pilot.sh"

for selection in metric random; do
  tmux new-session -d -s "v10m_${SIDE}_${CONDITION}_${selection}" env \
    REPO="${REPO}" SELECTION="${selection}" SIDE="${SIDE}" CONDITION="${CONDITION}" \
    RUN_ROOT="${RUN_ROOT}/${selection}" DATASET_PATH="${DATASET_PATH}" MODEL_PATH="${MODEL_PATH}" \
    INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH}" \
    TRACE_MANIFEST="${RUN_ROOT}/${selection}/replay/staged_manifest.json" \
    V10_GPU_POOL="${V10_GPU_POOL}" \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_metric_policy.sh"
done

for branch in metric random control; do
  session="v10m_${SIDE}_${CONDITION}_${branch}"
  marker="${RUN_ROOT}/${branch}/completed.txt"
  while [[ ! -s "${marker}" ]]; do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "Metric-pilot branch exited without completion: ${branch}" >&2
      exit 2
    fi
    sleep 30
  done
done
"${REPO}/.venv/bin/python" \
  "${REPO}/scripts/reinforcement_learning/rwm_trace/summarize_v10_metric_pilot.py" \
  --run-root "${RUN_ROOT}" --output "${RUN_ROOT}/comparison.json"
'
echo "Started ${controller}; it will prepare one shared candidate set and run metric/random/control."
