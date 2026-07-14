#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${1:?experiment root}"
JOBS_FILE="${EXPERIMENT_ROOT}/queues/jobs.list"
QUEUE_LOG="${EXPERIMENT_ROOT}/queues/queue_events.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_stage_job.sh"

GPU_POOL="${GPU_POOL:-0 1 2 3 4 5 6 7}"
GPU_MAX_MEMORY_USED_MIB="${GPU_MAX_MEMORY_USED_MIB:-6000}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RETRY_FAILED="${RETRY_FAILED:-false}"
POLICY_JOBS_PER_GPU="${POLICY_JOBS_PER_GPU:-2}"
POLICY_SECOND_SLOT_MAX_MEMORY_USED_MIB="${POLICY_SECOND_SLOT_MAX_MEMORY_USED_MIB:-16000}"

declare -A JOB_DIRS JOB_DEPS JOB_PIDS JOB_GPUS GPU_RUNNING_COUNT GPU_RUNNING_NONPOLICY
JOB_ORDER=()

while IFS='|' read -r job_id job_dir dependencies; do
  [[ -z "${job_id}" || "${job_id}" == \#* ]] && continue
  JOB_ORDER+=("${job_id}")
  JOB_DIRS["${job_id}"]="${job_dir}"
  JOB_DEPS["${job_id}"]="${dependencies}"
done < "${JOBS_FILE}"

status_of() {
  local job_id="$1"
  local state_file="${JOB_DIRS[${job_id}]}/state.txt"
  if [[ ! -f "${state_file}" ]]; then
    printf 'pending\n'
    return
  fi
  awk -F= '$1=="status" {print $2; exit}' "${state_file}"
}

write_terminal_state() {
  local job_id="$1"
  local status="$2"
  local detail="$3"
  local state_file="${JOB_DIRS[${job_id}]}/state.txt"
  local tmp="${state_file}.tmp.$$"
  {
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'job=%s\n' "${job_id}"
    printf 'status=%s\n' "${status}"
    printf 'pid=\n'
    printf 'gpu=\n'
    printf 'detail=%s\n' "${detail}"
  } > "${tmp}"
  mv "${tmp}" "${state_file}"
}

queue_event() {
  printf '[%s] scheduler %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${QUEUE_LOG}"
}

dependencies_ready() {
  local job_id="$1"
  local dependencies="${JOB_DEPS[${job_id}]}"
  [[ -z "${dependencies}" ]] && return 0
  local dependency status
  IFS=',' read -r -a dependency_values <<< "${dependencies}"
  for dependency in "${dependency_values[@]}"; do
    status="$(status_of "${dependency}")"
    if [[ "${status}" == "failed" || "${status}" == "blocked" ]]; then
      write_terminal_state "${job_id}" blocked "dependency=${dependency}:${status}"
      queue_event "job=${job_id} blocked dependency=${dependency}:${status}"
      return 1
    fi
    [[ "${status}" == "completed" ]] || return 1
  done
  return 0
}

refresh_workers() {
  local job_id pid gpu status
  GPU_RUNNING_COUNT=()
  GPU_RUNNING_NONPOLICY=()
  for job_id in "${JOB_ORDER[@]}"; do
    status="$(status_of "${job_id}")"
    if [[ "${status}" == "running" ]]; then
      pid="$(awk -F= '$1=="pid" {print $2; exit}' "${JOB_DIRS[${job_id}]}/state.txt")"
      gpu="$(awk -F= '$1=="gpu" {print $2; exit}' "${JOB_DIRS[${job_id}]}/state.txt")"
      if [[ -n "${pid}" && -e "/proc/${pid}" ]]; then
        JOB_PIDS["${job_id}"]="${pid}"
        JOB_GPUS["${job_id}"]="${gpu}"
        GPU_RUNNING_COUNT["${gpu}"]=$(( ${GPU_RUNNING_COUNT[${gpu}]:-0} + 1 ))
        if [[ "${job_id}" != policy_* ]]; then
          GPU_RUNNING_NONPOLICY["${gpu}"]=1
        fi
      elif [[ -n "${JOB_PIDS[${job_id}]:-}" ]]; then
        unset "JOB_PIDS[$job_id]"
        unset "JOB_GPUS[$job_id]"
      else
        write_terminal_state "${job_id}" failed "stale_running_state"
        queue_event "job=${job_id} failed stale_running_state"
      fi
    else
      unset "JOB_PIDS[$job_id]"
      unset "JOB_GPUS[$job_id]"
    fi
  done
}

available_gpus() {
  local job_id="$1"
  local allowed=" ${GPU_POOL} "
  local running_count running_nonpolicy memory_limit utilization_limit
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | while IFS=',' read -r index memory utilization; do
        index="${index//[[:space:]]/}"
        memory="${memory//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${allowed}" == *" ${index} "* ]] || continue
        running_count="${GPU_RUNNING_COUNT[${index}]:-0}"
        running_nonpolicy="${GPU_RUNNING_NONPOLICY[${index}]:-0}"
        memory_limit="${GPU_MAX_MEMORY_USED_MIB}"
        utilization_limit="${GPU_MAX_UTILIZATION}"
        if [[ "${job_id}" == policy_* ]]; then
          (( running_nonpolicy == 0 && running_count < POLICY_JOBS_PER_GPU )) || continue
          if (( running_count > 0 )); then
            memory_limit="${POLICY_SECOND_SLOT_MAX_MEMORY_USED_MIB}"
            utilization_limit=100
          fi
        else
          (( running_count == 0 )) || continue
        fi
        if (( memory <= memory_limit && utilization <= utilization_limit )); then
          printf '%012d %012d %s\n' "${memory}" "${utilization}" "${index}"
        fi
      done \
    | sort -n \
    | awk '{print $3}'
}

start_job() {
  local job_id="$1"
  local gpu="$2"
  local job_dir="${JOB_DIRS[${job_id}]}"
  local command_file="${job_dir}/command.sh"
  mkdir -p "${job_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${WORKER}" "${job_id}" "${job_dir}" "${command_file}" "${QUEUE_LOG}" &
  local pid=$!
  JOB_PIDS["${job_id}"]="${pid}"
  JOB_GPUS["${job_id}"]="${gpu}"
  GPU_RUNNING_COUNT["${gpu}"]=$(( ${GPU_RUNNING_COUNT[${gpu}]:-0} + 1 ))
  if [[ "${job_id}" != policy_* ]]; then
    GPU_RUNNING_NONPOLICY["${gpu}"]=1
  fi
  queue_event "job=${job_id} dispatched pid=${pid} gpu=${gpu}"
}

queue_event "started root=${EXPERIMENT_ROOT} gpu_pool=${GPU_POOL} policy_jobs_per_gpu=${POLICY_JOBS_PER_GPU}"
while true; do
  refresh_workers

  for job_id in "${JOB_ORDER[@]}"; do
    status="$(status_of "${job_id}")"
    if [[ "${status}" == "failed" && "${RETRY_FAILED}" == "true" ]]; then
      write_terminal_state "${job_id}" pending "manual_retry_enabled"
      status=pending
    fi
    [[ "${status}" == "pending" || "${status}" == "dry_run" ]] || continue
    dependencies_ready "${job_id}" || continue
    mapfile -t free_gpus < <(available_gpus "${job_id}")
    ((${#free_gpus[@]} > 0)) || continue
    start_job "${job_id}" "${free_gpus[0]}"
  done

  terminal=0
  completed=0
  failed=0
  blocked=0
  for job_id in "${JOB_ORDER[@]}"; do
    status="$(status_of "${job_id}")"
    case "${status}" in
      completed) completed=$((completed + 1)); terminal=$((terminal + 1)) ;;
      failed) failed=$((failed + 1)); terminal=$((terminal + 1)) ;;
      blocked) blocked=$((blocked + 1)); terminal=$((terminal + 1)) ;;
    esac
  done
  if (( terminal == ${#JOB_ORDER[@]} )); then
    queue_event "finished completed=${completed} failed=${failed} blocked=${blocked}"
    printf 'completed=%d\nfailed=%d\nblocked=%d\ntotal=%d\n' \
      "${completed}" "${failed}" "${blocked}" "${#JOB_ORDER[@]}" \
      > "${EXPERIMENT_ROOT}/queues/final_state.txt"
    [[ "${failed}" -eq 0 && "${blocked}" -eq 0 ]]
    exit
  fi
  sleep "${POLL_SECONDS}"
done
