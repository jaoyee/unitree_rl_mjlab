#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PAYLOAD_ROOT="${PAYLOAD_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/payload_20260713_payload_0_vs_5to10kg}"
RR_ROOT="${RR_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/rr_calf_20260713_rrcalf_1_vs_0p5to1}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
SESSION_PREFIX="${SESSION_PREFIX:-go2_gap_expert}"
QUEUE_LOG="${REPO_ROOT}/logs/experiments/go2_gap_experts/launcher_events.log"

mkdir -p "$(dirname "${QUEUE_LOG}")"

run_job() {
  local gpu="$1" root="$2" job="$3"
  local stage_dir="${root}/${job}"
  local summary="${stage_dir}/summary.json"
  local command_file="${root}/commands/${job}.sh"
  if [[ -f "${summary}" ]] && grep -q '"status": "completed"' "${summary}"; then
    printf '[%s] skip completed gpu=%s job=%s\n' "$(date -Is)" "${gpu}" "${job}" | tee -a "${QUEUE_LOG}"
    return
  fi
  local free_gb
  free_gb="$(df -BG --output=avail /data1 | tail -1 | tr -dc '0-9')"
  if (( free_gb < MIN_FREE_GB )); then
    printf '[%s] stop low_disk free_gb=%s job=%s\n' "$(date -Is)" "${free_gb}" "${job}" | tee -a "${QUEUE_LOG}"
    return 75
  fi
  printf '[%s] start gpu=%s job=%s free_gb=%s\n' "$(date -Is)" "${gpu}" "${job}" "${free_gb}" | tee -a "${QUEUE_LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash "${command_file}"
  printf '[%s] completed gpu=%s job=%s\n' "$(date -Is)" "${gpu}" "${job}" | tee -a "${QUEUE_LOG}"
}

worker() {
  local gpu="$1"
  shift
  local item root job
  for item in "$@"; do
    root="${item%%:*}"
    job="${item##*:}"
    run_job "${gpu}" "${root}" "${job}"
  done
}

launch_worker() {
  local gpu="$1"
  shift
  local session="${SESSION_PREFIX}_gpu${gpu}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    return 1
  fi
  local quoted=("$(printf '%q' "${BASH_SOURCE[0]}")" --worker "$(printf '%q' "${gpu}")")
  local item
  for item in "$@"; do
    quoted+=("$(printf '%q' "${item}")")
  done
  tmux new-session -d -s "${session}" "bash ${quoted[*]}"
  echo "launched ${session}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker "$@"
  exit 0
fi

# Four independent workers; each GPU executes three full expert jobs serially.
launch_worker 1 "${PAYLOAD_ROOT}:G0_D0" "${PAYLOAD_ROOT}:G1_D0" "${RR_ROOT}:G0_D0"
launch_worker 2 "${PAYLOAD_ROOT}:G0_D1" "${PAYLOAD_ROOT}:G1_D1" "${RR_ROOT}:G0_D1"
launch_worker 3 "${PAYLOAD_ROOT}:G0_D2" "${PAYLOAD_ROOT}:G1_D2" "${RR_ROOT}:G0_D2"
launch_worker 7 "${RR_ROOT}:G1_D0" "${RR_ROOT}:G1_D1" "${RR_ROOT}:G1_D2"

tmux ls | grep "${SESSION_PREFIX}" || true
