#!/usr/bin/env bash
set -euo pipefail

: "${V10_GPU_POOL:?Set comma-separated physical GPU IDs}"
MAX_USED_MIB="${V10_GPU_MAX_USED_MIB:-2048}"
POLL_SECONDS="${V10_GPU_POLL_SECONDS:-20}"
LOCK_ROOT="${V10_GPU_LOCK_ROOT:-/tmp/go2_trace_v10_gpu_locks}"
mkdir -p "${LOCK_ROOT}"

while true; do
  IFS=',' read -ra GPU_IDS <<< "${V10_GPU_POOL}"
  for gpu in "${GPU_IDS[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Bad GPU id ${gpu}" >&2; exit 2; }
    lock="${LOCK_ROOT}/gpu_${gpu}.lock"
    exec {fd}>"${lock}"
    if ! flock -n "${fd}"; then
      exec {fd}>&-
      continue
    fi
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d '[:space:]')"
    if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_USED_MIB )); then
      flock -u "${fd}"
      exec {fd}>&-
      continue
    fi
    echo "v10_gpu_stage physical_gpu=${gpu} initial_used_mib=${used} command=$*"
    status=0
    CUDA_VISIBLE_DEVICES="${gpu}" "$@" || status=$?
    flock -u "${fd}"
    exec {fd}>&-
    exit "${status}"
  done
  sleep "${POLL_SECONDS}"
done
