#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${V10_ARTIFACT_ENV:?Set the server-local V10 path manifest}"
: "${V10_RUN_BASE:?Set a new formal run base}"
: "${V10_GPU_POOL:?Set server-local physical GPU pool}"
: "${V10_PILOT_GATE:?Set the copied pilots_passed.env produced by V10 pilot evaluation}"
[[ -s "${V10_PILOT_GATE}" ]] && grep -qx 'passed=true' "${V10_PILOT_GATE}" || {
  echo "V10 formal launch is blocked until both pilots pass: ${V10_PILOT_GATE}" >&2; exit 2;
}
V10_JOBS="${V10_JOBS:-sim:g0 sim:rr05 sim:rr03 sim:p5 sim:p75 real:g0 real:rr05 real:rr03 real:p5 real:p75}"
source "${V10_ARTIFACT_ENV}"
mkdir -p "${V10_RUN_BASE}/initial_scorers"

value_for() {
  local side="$1" condition="$2" suffix="$3"
  local name="${side^^}_${condition^^}_${suffix}"
  printf '%s' "${!name:-}"
}

for job in ${V10_JOBS}; do
  side="${job%%:*}"; condition="${job##*:}"
  dataset="$(value_for "${side}" "${condition}" DATASET)"
  model="$(value_for "${side}" "${condition}" MODEL)"
  warmup="$(value_for "${side}" "${condition}" WARMUP)"
  [[ -s "${dataset}" && -s "${model}" && -s "${warmup}/actor.pt" ]] || {
    echo "Missing artifacts for ${job}" >&2; exit 2;
  }
  if [[ "${job}" == real:g0 && "${REAL_G0_SOURCE:-}" != go2sun_recollect_20260719 ]]; then
    echo "V10 real/g0 requires the new go2sun recollection, not legacy go2 data." >&2
    exit 2
  fi
  session="v10_${side}_${condition}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2; exit 2
  fi
  root="${V10_RUN_BASE}/${side}/${condition}"
  scorer="${V10_RUN_BASE}/initial_scorers/${side}_${condition}"
  tmux new-session -d -s "${session}" env \
    REPO="${REPO}" SIDE="${side}" CONDITION="${condition}" RUN_ROOT="${root}" \
    DATASET_PATH="${dataset}" MODEL_PATH="${model}" INITIAL_POLICY_PATH="${warmup}" \
    INITIAL_SCORER_ROOT="${scorer}" V10_GPU_POOL="${V10_GPU_POOL}" RUN_KIND=formal \
    CODEX_LOCK="${V10_RUN_BASE}/codex_feedback.lock" \
    bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_condition.sh"
  echo "started ${session}"

  baseline_name="${side^^}_${condition^^}_BASELINE_MANIFEST"
  baseline_manifest="${!baseline_name:-${V10_RUN_BASE}/baselines/${side}/${condition}/v10_baseline_manifest.json}"
  baseline_valid=false
  if [[ -s "${baseline_manifest}" ]]; then
    if "${REPO}/.venv/bin/python" \
      "${REPO}/scripts/reinforcement_learning/rwm_trace/validate_v10_baseline.py" \
      --manifest "${baseline_manifest}" \
      --protocol "${REPO}/scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json" \
      --side "${side}" --condition "${condition}" --dataset "${dataset}" --model "${model}" \
      --warmup "${warmup}" --reward-source "${REPO}/src/tasks/rwm_velocity/mdp/rewards.py" \
      --training-steps 40000000; then
      baseline_valid=true
    fi
  fi
  if [[ "${baseline_valid}" != true ]]; then
    baseline_root="${V10_RUN_BASE}/baselines/${side}/${condition}"
    baseline_session="v10_base_${side}_${condition}"
    tmux new-session -d -s "${baseline_session}" env \
      REPO="${REPO}" SIDE="${side}" CONDITION="${condition}" RUN_ROOT="${baseline_root}" RUN_KIND=formal \
      DATASET_PATH="${dataset}" MODEL_PATH="${model}" INITIAL_POLICY_PATH="${warmup}" \
      V10_GPU_POOL="${V10_GPU_POOL}" \
      bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v10_no_trace_control.sh"
    echo "started ${baseline_session}"
  else
    canonical_baseline_root="${V10_RUN_BASE}/baselines/${side}/${condition}"
    mkdir -p "${canonical_baseline_root}"
    cp --reflink=auto "${baseline_manifest}" "${canonical_baseline_root}/v10_baseline_manifest.json.new"
    mv "${canonical_baseline_root}/v10_baseline_manifest.json.new" \
      "${canonical_baseline_root}/v10_baseline_manifest.json"
  fi
done
