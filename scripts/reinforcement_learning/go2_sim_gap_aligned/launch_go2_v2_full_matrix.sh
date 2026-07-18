#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${DOMAIN:?Set DOMAIN to sim or real}"
: "${DATASET_ROOT:?Set logs/rwm_datasets_v2/{sim,real}_v2}"
: "${GPU_QUEUES:?Example: 2:g0,rr03;5:rr05,p5;1:p75}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/logs/experiments/go2_v2_contact18_matched25k/${DOMAIN}/${RUN_ID}}"
mkdir -p "${EXPERIMENT_ROOT}"

IFS=';' read -ra QUEUES <<< "${GPU_QUEUES}"
queue_index=0
for queue in "${QUEUES[@]}"; do
  gpu="${queue%%:*}"
  conditions_csv="${queue#*:}"
  [[ -n "${gpu}" && -n "${conditions_csv}" && "${queue}" == *:* ]] || { echo "Bad queue: ${queue}" >&2; exit 2; }
  session="go2_v2_${DOMAIN}_q${queue_index}_${RUN_ID}"
  command_file="${EXPERIMENT_ROOT}/queue_${queue_index}_gpu${gpu}.sh"
  IFS=',' read -ra conditions <<< "${conditions_csv}"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${REPO_ROOT}"
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${gpu}"
    echo 'failures=0'
    for condition in "${conditions[@]}"; do
      dataset="${DATASET_ROOT}/${condition}/dataset.pt"
      run_root="${EXPERIMENT_ROOT}/${condition}"
      printf 'echo "[queue] start %s at $(date --iso-8601=seconds)"\n' "${condition}"
      printf 'if ! DOMAIN=%q GAP_ID=%q DATASET_PATH=%q RUN_ROOT=%q bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_v2_full_pipeline.sh; then echo "[queue] failed %s at $(date --iso-8601=seconds)"; failures=$((failures + 1)); fi\n' \
        "${DOMAIN}" "${condition}" "${dataset}" "${run_root}" "${condition}"
    done
    echo 'exit "${failures}"'
  } > "${command_file}"
  chmod +x "${command_file}"
  tmux new-session -d -s "${session}" "bash '${command_file}' 2>&1 | tee -a '${EXPERIMENT_ROOT}/queue_${queue_index}.log'"
  printf 'queue=%s gpu=%s conditions=%s session=%s command=%s\n' \
    "${queue_index}" "${gpu}" "${conditions_csv}" "${session}" "${command_file}" \
    | tee -a "${EXPERIMENT_ROOT}/sessions.txt"
  queue_index=$((queue_index + 1))
done

printf '%s\n' "${EXPERIMENT_ROOT}" > "${REPO_ROOT}/logs/experiments/go2_v2_contact18_matched25k/latest_${DOMAIN}_root.txt"
echo "experiment_root=${EXPERIMENT_ROOT}"
