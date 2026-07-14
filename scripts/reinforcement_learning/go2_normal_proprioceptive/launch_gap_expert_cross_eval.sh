#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PAYLOAD_ROOT="${PAYLOAD_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/payload_20260713_payload_0_vs_5to10kg}"
RR_ROOT="${RR_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/rr_calf_20260713_rrcalf_1_vs_0p5to1}"
QUEUE_ROOT="${QUEUE_ROOT:-${REPO_ROOT}/logs/experiments/go2_gap_experts/cross_eval_20260713}"
NUM_ENVS="${NUM_ENVS:-256}"
STEPS="${STEPS:-2400}"
SEED="${SEED:-400}"
GPU_CSV="${GPU_CSV:-1,2,3,4}"
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"

GAP_KIND=payload EXPERIMENT_ROOT="${PAYLOAD_ROOT}" \
  NUM_ENVS="${NUM_ENVS}" STEPS="${STEPS}" SEED="${SEED}" \
  bash "${SCRIPT_DIR}/prepare_gap_expert_cross_eval.sh"
GAP_KIND=rr_calf EXPERIMENT_ROOT="${RR_ROOT}" \
  NUM_ENVS="${NUM_ENVS}" STEPS="${STEPS}" SEED="${SEED}" \
  bash "${SCRIPT_DIR}/prepare_gap_expert_cross_eval.sh"

mkdir -p "${QUEUE_ROOT}"
cat "${PAYLOAD_ROOT}/cross_eval/jobs.list" "${RR_ROOT}/cross_eval/jobs.list" \
  > "${QUEUE_ROOT}/all.jobs"
for gpu in "${GPUS[@]}"; do
  : > "${QUEUE_ROOT}/gpu_${gpu}.jobs"
done

index=0
while IFS= read -r line; do
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  printf '%s\n' "${line}" >> "${QUEUE_ROOT}/gpu_${gpu}.jobs"
  index=$((index + 1))
done < "${QUEUE_ROOT}/all.jobs"

cat > "${QUEUE_ROOT}/manifest.txt" <<EOF
created_at=$(date -Is)
payload_root=${PAYLOAD_ROOT}
rr_root=${RR_ROOT}
num_envs=${NUM_ENVS}
steps=${STEPS}
seed=${SEED}
gpus=${GPU_CSV}
jobs=${index}
EOF

for gpu in "${GPUS[@]}"; do
  session="go2_gap_eval_gpu${gpu}"
  jobs_file="${QUEUE_ROOT}/gpu_${gpu}.jobs"
  worker_dir="${QUEUE_ROOT}/gpu_${gpu}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${worker_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' JOBS_FILE='${jobs_file}' WORKER_DIR='${worker_dir}' bash '${SCRIPT_DIR}/run_gap_cross_eval_worker.sh' 2>&1 | tee -a '${worker_dir}/worker.log'"
  echo "started ${session} jobs=$(wc -l < "${jobs_file}")"
done

echo "queue_root=${QUEUE_ROOT}"
