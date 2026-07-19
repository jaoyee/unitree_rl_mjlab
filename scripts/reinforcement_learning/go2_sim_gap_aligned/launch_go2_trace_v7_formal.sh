#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${V7_DATASET_MANIFEST:?Set V7_DATASET_MANIFEST to the validated 10-dataset env file}"
[[ -s "${V7_DATASET_MANIFEST}" ]] || { echo "Missing ${V7_DATASET_MANIFEST}" >&2; exit 2; }
# shellcheck disable=SC1090
source "${V7_DATASET_MANIFEST}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FORMAL_ROOT="${FORMAL_ROOT:-${REPO}/logs/experiments/go2_trace_v7_corrected/${RUN_ID}}"
REUSE_ROOT="${REUSE_ROOT:-}"
mkdir -p "${FORMAL_ROOT}/logs"

# GPU 2 has insufficient free memory and GPU 0 is reserved.  The second job
# assigned to a GPU waits on flock until the first condition fully completes.
declare -A GPU=(
  [sim_g0]="${GPU_SIM_G0:-1}" [sim_rr05]="${GPU_SIM_RR05:-3}"
  [sim_p5]="${GPU_SIM_P5:-4}" [sim_rr03]="${GPU_SIM_RR03:-5}"
  [sim_p75]="${GPU_SIM_P75:-6}" [real_g0]="${GPU_REAL_G0:-7}"
  [real_rr05]="${GPU_REAL_RR05:-1}" [real_p5]="${GPU_REAL_P5:-3}"
  [real_rr03]="${GPU_REAL_RR03:-4}" [real_p75]="${GPU_REAL_P75:-5}"
)

for side in sim real; do
  for condition in g0 rr05 p5 rr03 p75; do
    key="${side}_${condition}"
    gpu="${GPU[$key]}"
    session="v7f_${key}"
    run_root="${FORMAL_ROOT}/${side}/${condition}"
    mkdir -p "${run_root}"

    dataset_var="${side^^}_${condition^^}_DATASET"
    if [[ -v "${dataset_var}" ]]; then
      dataset_path="${!dataset_var}"
    else
      dataset_path=""
    fi
    [[ -s "${dataset_path}" ]] || { echo "Missing ${dataset_var}: ${dataset_path}" >&2; exit 2; }

    if [[ -n "${REUSE_ROOT}" ]]; then
      source_root="${REUSE_ROOT}/${side}/${condition}"
      for name in ready.env scorer_ready.env prepare_completed.txt; do
        [[ -s "${source_root}/${name}" ]] || { echo "Missing reusable ${source_root}/${name}" >&2; exit 2; }
        cp -a --reflink=auto "${source_root}/${name}" "${run_root}/${name}"
      done
      printf 'status=reused_immutable_inputs_only\nsource=%s\ntime=%s\n' \
        "${source_root}" "$(date --iso-8601=seconds)" > "${run_root}/reuse_manifest.txt"
    fi

    command_file="${run_root}/formal_command.sh"
    cat > "${command_file}.new" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec 9>"/tmp/v7f_gpu${gpu}.lock"
if ! flock -n 9; then
  printf 'time=%s\\nside=${side}\\ncondition=${condition}\\nstage=gpu_queue\\nstatus=waiting\\ngpu=${gpu}\\n' \
    "\$(date --iso-8601=seconds)" > "${run_root}/formal_state.txt"
  flock 9
fi
printf 'time=%s\\nside=${side}\\ncondition=${condition}\\nstage=formal_pipeline\\nstatus=running\\ngpu=${gpu}\\n' \
  "\$(date --iso-8601=seconds)" > "${run_root}/formal_state.txt"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
REPO="${REPO}" SIDE="${side}" CONDITION="${condition}" \
DATASET_PATH="${dataset_path}" RUN_ROOT="${run_root}" CUDA_VISIBLE_DEVICES="${gpu}" \
TRACE_T=2 TRACE_H=100 SELECT_RATIO=0.25 REFRESH_CYCLES=8 SEGMENT_ENV_STEPS=5000000 \
bash scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_trace_v7_condition.sh \
  2>&1 | tee -a "${run_root}/formal.log"
printf 'time=%s\\nside=${side}\\ncondition=${condition}\\nstage=done\\nstatus=completed\\ngpu=${gpu}\\n' \
  "\$(date --iso-8601=seconds)" > "${run_root}/formal_state.txt"
EOF
    bash -n "${command_file}.new"
    mv "${command_file}.new" "${command_file}"
    chmod +x "${command_file}"

    if ! tmux has-session -t "${session}" 2>/dev/null; then
      tmux new-session -d -s "${session}" "bash '${command_file}'"
    fi
  done
done

cat > "${FORMAL_ROOT}/formal_manifest.txt" <<EOF
protocol=corrected_v7_default_formal
trace_t=2
trace_h=100
select_ratio=0.25
refresh_cycles=8
segment_env_steps=5000000
common_warmup_steps=10000000
total_policy_budget=50000000
trajectories_per_state=4
num_dataset_states=1024
branches=r10,r25
failure_quota_mode=disabled_natural
dataset_manifest=${V7_DATASET_MANIFEST}
reuse_root=${REUSE_ROOT}
EOF

tmux list-sessions -F '#{session_name}' | grep '^v7f_' | sort
