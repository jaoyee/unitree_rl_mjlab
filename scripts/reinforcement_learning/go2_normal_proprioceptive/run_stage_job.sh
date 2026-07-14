#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:?job id}"
JOB_DIR="${2:?job dir}"
COMMAND_FILE="${3:?command file}"
QUEUE_LOG="${4:?queue log}"
GPU_ID="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must contain one physical GPU id}"
STATE_FILE="${JOB_DIR}/state.txt"

write_state() {
  local status="$1"
  local detail="${2:-}"
  local tmp="${STATE_FILE}.tmp.$$"
  {
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'job=%s\n' "${JOB_ID}"
    printf 'status=%s\n' "${status}"
    printf 'pid=%s\n' "$$"
    printf 'gpu=%s\n' "${GPU_ID}"
    printf 'detail=%s\n' "${detail}"
  } > "${tmp}"
  mv "${tmp}" "${STATE_FILE}"
}

queue_event() {
  printf '[%s] job=%s gpu=%s %s\n' "$(date --iso-8601=seconds)" "${JOB_ID}" "${GPU_ID}" "$*" \
    | tee -a "${QUEUE_LOG}"
}

on_exit() {
  local code=$?
  trap - EXIT
  if [[ "${code}" -ne 0 ]]; then
    write_state failed "exit_code=${code}"
    queue_event "failed exit_code=${code}"
  fi
  exit "${code}"
}
trap on_exit EXIT

mkdir -p "${JOB_DIR}"
write_state running "command=${COMMAND_FILE}"
queue_event "started"

set +e
bash "${COMMAND_FILE}" 2>&1 | tee "${JOB_DIR}/run.log"
command_status=${PIPESTATUS[0]}
set -e
if [[ "${command_status}" -ne 0 ]]; then
  exit "${command_status}"
fi

if [[ ! -s "${JOB_DIR}/summary.json" ]]; then
  echo "Stage did not produce ${JOB_DIR}/summary.json" >&2
  exit 70
fi
python_bin="${PYTHON_BIN:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.venv/bin/python}"
"${python_bin}" - "${JOB_DIR}/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("status") != "completed":
    raise SystemExit(f"stage summary is not completed: {summary!r}")
PY

write_state completed "summary=${JOB_DIR}/summary.json"
queue_event "completed"
trap - EXIT
