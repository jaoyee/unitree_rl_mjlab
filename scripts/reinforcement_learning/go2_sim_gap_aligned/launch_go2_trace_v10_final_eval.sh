#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${V10_ARTIFACT_ENV:?Set artifact manifest}"
: "${V10_RUN_BASE:?Set formal run base}"
: "${V10_EVAL_ROOT:?Set final evaluation root}"
: "${V10_GPU_POOL:?Set GPU pool}"
V10_JOBS="${V10_JOBS:-sim:g0 sim:rr05 sim:rr03 sim:p5 sim:p75 real:g0 real:rr05 real:rr03 real:p5 real:p75}"
source "${V10_ARTIFACT_ENV}"
GPU_EXEC="${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_v10_gpu_stage.sh"
mkdir -p "${V10_EVAL_ROOT}/commands"
for job in ${V10_JOBS}; do
  side="${job%%:*}"; condition="${job##*:}"
  model_name="${side^^}_${condition^^}_MODEL"; model="${!model_name}"
  trace_root="${V10_RUN_BASE}/${side}/${condition}"
  [[ -s "${trace_root}/completed.txt" ]] || { echo "TRACE incomplete: ${job}" >&2; exit 2; }
  trace_policy="$(<"${trace_root}/refresh_08/policy/stage/artifact_path.txt")"
  baseline_manifest="${V10_RUN_BASE}/baselines/${side}/${condition}/v10_baseline_manifest.json"
  baseline_policy="$(${REPO}/.venv/bin/python - "${baseline_manifest}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["policy"])
PY
)"
  for variant in baseline trace; do
    checkpoint="${trace_policy}"; [[ "${variant}" == baseline ]] && checkpoint="${baseline_policy}"
    V10_GPU_POOL="${V10_GPU_POOL}" "${GPU_EXEC}" env REPO="${REPO}" \
      CHECKPOINT_PATH="${checkpoint}" MODEL_PATH="${model}" CONDITION="${condition}" \
      OUTPUT_DIR="${V10_EVAL_ROOT}/gap_only/${side}/${condition}/${variant}" \
      COMMAND_ROOT="${V10_EVAL_ROOT}/commands" NUM_ENVS=256 STEPS=1000 \
      SEEDS='400 401 402' SWITCH_STEPS=50 \
      bash "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/run_go2_v7_random50_one.sh"
  done
done
"${REPO}/.venv/bin/python" \
  "${REPO}/scripts/reinforcement_learning/go2_sim_gap_aligned/summarize_go2_trace_v7_random50_eval.py" \
  --eval-root "${V10_EVAL_ROOT}"
