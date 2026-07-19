#!/usr/bin/env bash
set -euo pipefail

: "${WORKER_COMMAND:?}"
: "${FORMAL_ROOT:?Set FORMAL_ROOT to the active V7 formal experiment root}"
INITIAL_DELAY="${INITIAL_DELAY:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MAX_UTIL="${MAX_UTIL:-20}"
sleep "${INITIAL_DELAY}"

is_formal_reserved() {
  local gpu="$1" state status assigned
  while IFS= read -r state; do
    status="$(sed -n 's/^status=//p' "${state}" | tail -1)"
    assigned="$(sed -n 's/^gpu=//p' "${state}" | tail -1)"
    if [[ "${status}" != completed && "${assigned}" == "${gpu}" ]]; then return 0; fi
  done < <(find "${FORMAL_ROOT}" -name state.txt -type f 2>/dev/null)
  return 1
}

while true; do
  for gpu in 1 2 3 4 5 6 7; do
    is_formal_reserved "${gpu}" && continue
    line="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "${gpu}")"
    free="${line%%,*}"; util="${line##*,}"
    free="${free// /}"; util="${util// /}"
    [[ "${free}" -ge "${MIN_FREE_MIB}" && "${util}" -le "${MAX_UTIL}" ]] || continue
    lock="/tmp/go2_v7_gpu_${gpu}.lock"
    if mkdir "${lock}" 2>/dev/null; then
      trap 'rm -rf -- "${lock}"' EXIT INT TERM
      export CUDA_VISIBLE_DEVICES="${gpu}"
      echo "[gpu-acquired] gpu=${gpu} free_mib=${free} util=${util} time=$(date --iso-8601=seconds)"
      status=0
      bash "${WORKER_COMMAND}" || status=$?
      rm -rf -- "${lock}"
      trap - EXIT INT TERM
      exit "${status}"
    fi
  done
  echo "[gpu-wait] no eligible GPU at $(date --iso-8601=seconds)"
  sleep 60
done
