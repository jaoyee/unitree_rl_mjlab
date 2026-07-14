#!/usr/bin/env bash
set -uo pipefail

: "${JOBS_FILE:?Set JOBS_FILE}"
: "${WORKER_DIR:?Set WORKER_DIR}"

mkdir -p "${WORKER_DIR}/done" "${WORKER_DIR}/failed"
QUEUE_LOG="${WORKER_DIR}/queue_events.log"
failures=0

while IFS='|' read -r job command_file; do
  [[ -n "${job}" && -n "${command_file}" ]] || continue
  if [[ -f "${WORKER_DIR}/done/${job}" ]]; then
    printf '[%s] skip completed %s\n' "$(date -Is)" "${job}" | tee -a "${QUEUE_LOG}"
    continue
  fi
  printf '[%s] start %s command=%s\n' "$(date -Is)" "${job}" "${command_file}" | tee -a "${QUEUE_LOG}"
  if bash "${command_file}"; then
    date -Is > "${WORKER_DIR}/done/${job}"
    rm -f "${WORKER_DIR}/failed/${job}"
    printf '[%s] completed %s\n' "$(date -Is)" "${job}" | tee -a "${QUEUE_LOG}"
  else
    code=$?
    printf '%s exit_code=%d\n' "$(date -Is)" "${code}" > "${WORKER_DIR}/failed/${job}"
    printf '[%s] failed %s exit_code=%d\n' "$(date -Is)" "${job}" "${code}" | tee -a "${QUEUE_LOG}"
    failures=$((failures + 1))
  fi
done < "${JOBS_FILE}"

printf '[%s] worker finished failures=%d\n' "$(date -Is)" "${failures}" | tee -a "${QUEUE_LOG}"
exit "${failures}"
