#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${V10_ARTIFACT_ENV:?Set server-local V10 path manifest}"
: "${V10_RUN_BASE:?Set a new pilot run base}"
: "${V10_GPU_POOL:?Set physical GPU pool}"
source "${V10_ARTIFACT_ENV}"
mkdir -p "${V10_RUN_BASE}/initial_scorers"

launch_condition() {
  local condition="$1" kind="$2" session="$3" root="$4"
  local prefix="REAL_${condition^^}"
  local dataset_var="${prefix}_DATASET" model_var="${prefix}_MODEL" warmup_var="${prefix}_WARMUP"
  local dataset="${!dataset_var:-}" model="${!model_var:-}" warmup="${!warmup_var:-}"
  local scorer="${V10_RUN_BASE}/initial_scorers/real_${condition}"
  [[ -s "${dataset}" && -s "${model}" && -s "${warmup}/actor.pt" ]] || {
    echo "Missing V10 pilot artifacts for real/${condition}" >&2; exit 2;
  }
  tmux has-session -t "${session}" 2>/dev/null && {
    echo "Session already exists: ${session}" >&2; exit 2;
  }
  tmux new-session -d -s "${session}" env \
    REPO="${REPO}" SIDE=real CONDITION="${condition}" RUN_ROOT="${root}" \
    DATASET_PATH="${dataset}" MODEL_PATH="${model}" INITIAL_POLICY_PATH="${warmup}" \
    INITIAL_SCORER_ROOT="${scorer}" V10_GPU_POOL="${V10_GPU_POOL}" RUN_KIND="${kind}" \
    CODEX_LOCK="${V10_RUN_BASE}/codex_feedback.lock" \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_condition.sh"
}

for condition in rr05 p5; do
  root="${V10_RUN_BASE}/trace/real/${condition}"
  if [[ ! -s "${root}/preflight_passed.env" ]]; then
    launch_condition "${condition}" preflight "v10_preflight_real_${condition}" "${root}"
  fi
done

controller="v10_pilot_controller"
tmux has-session -t "${controller}" 2>/dev/null && {
  echo "Session already exists: ${controller}" >&2; exit 2;
}
tmux new-session -d -s "${controller}" env \
  REPO="${REPO}" V10_ARTIFACT_ENV="${V10_ARTIFACT_ENV}" \
  V10_RUN_BASE="${V10_RUN_BASE}" V10_GPU_POOL="${V10_GPU_POOL}" \
  bash -lc '
set -euo pipefail
source "${V10_ARTIFACT_ENV}"
for condition in rr05 p5; do
  marker="${V10_RUN_BASE}/trace/real/${condition}/preflight_passed.env"
  while [[ ! -s "${marker}" ]]; do
    if ! tmux has-session -t "v10_preflight_real_${condition}" 2>/dev/null; then
      echo "Preflight exited without passing: real/${condition}" >&2
      exit 2
    fi
    sleep 30
  done
  grep -qx "passed=true" "${marker}" || exit 2
  expected_sha="$(sha256sum "${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json" | awk "{print \$1}")"
  grep -qx "protocol_sha256=${expected_sha}" "${marker}" || {
    echo "Stale V10 preflight protocol: ${marker}" >&2; exit 2;
  }
done
for condition in rr05 p5; do
  prefix="REAL_${condition^^}"
  dataset_var="${prefix}_DATASET"; model_var="${prefix}_MODEL"; warmup_var="${prefix}_WARMUP"
  dataset="${!dataset_var}"; model="${!model_var}"; warmup="${!warmup_var}"
  scorer="${V10_RUN_BASE}/initial_scorers/real_${condition}"
  trace_root="${V10_RUN_BASE}/trace/real/${condition}"
  control_root="${V10_RUN_BASE}/control/real/${condition}"
  tmux new-session -d -s "v10_pilot_real_${condition}" env \
    REPO="${REPO}" SIDE=real CONDITION="${condition}" RUN_ROOT="${trace_root}" \
    DATASET_PATH="${dataset}" MODEL_PATH="${model}" INITIAL_POLICY_PATH="${warmup}" \
    INITIAL_SCORER_ROOT="${scorer}" V10_GPU_POOL="${V10_GPU_POOL}" RUN_KIND=pilot \
    CODEX_LOCK="${V10_RUN_BASE}/codex_feedback.lock" \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_condition.sh"
  tmux new-session -d -s "v10_control_real_${condition}" env \
    REPO="${REPO}" SIDE=real CONDITION="${condition}" RUN_ROOT="${control_root}" RUN_KIND=pilot \
    DATASET_PATH="${dataset}" MODEL_PATH="${model}" INITIAL_POLICY_PATH="${warmup}" \
    V10_GPU_POOL="${V10_GPU_POOL}" \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_no_trace_control.sh"
done
'
echo "V10 candidate-only preflights started; ${controller} will launch both paired pilots after both pass."
