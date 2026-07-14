#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${STAGE_DIR:?Set STAGE_DIR}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-256}"
COMMAND_SWITCH_STEPS="${COMMAND_SWITCH_STEPS:-300}"
SEED="${SEED:-400}"
STEPS="${STEPS:-2400}"
COMMANDS=(
  "0.0,0.0,0.0"
  "0.5,0.0,0.0"
  "-0.4,0.0,0.0"
  "0.0,0.25,0.0"
  "0.0,-0.25,0.0"
  "0.0,0.0,0.5"
  "0.0,0.0,-0.5"
  "0.4,0.2,0.3"
)

if [[ ! -f "${CHECKPOINT_PATH}/actor.pt" ]]; then
  echo "Policy actor not found: ${CHECKPOINT_PATH}/actor.pt" >&2
  exit 1
fi
mkdir -p "${STAGE_DIR}/domains"

run_domain() {
  local domain="$1" clean_flag="$2" preset="$3" scale="$4"
  local output="${STAGE_DIR}/domains/${domain}.json"
  local command_args=()
  local command
  for command in "${COMMANDS[@]}"; do
    command_args+=("--command_sequence=${command}")
  done
  "${PYTHON_BIN}" scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --task Unitree-Go2-Flat-Normal-FixStand-RWM-Pretrain-Ens \
    --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" \
    --steps "${STEPS}" \
    --seed "${SEED}" \
    "${clean_flag}" \
    --randomization_preset "${preset}" \
    --randomization_components all \
    --randomization_scale "${scale}" \
    "${command_args[@]}" \
    --command_switch_steps "${COMMAND_SWITCH_STEPS}" \
    --output_json "${output}" \
    > "${STAGE_DIR}/domains/${domain}.log" 2>&1
}

run_domain V0_clean --clean default 0.0
run_domain V1_calibrated_default --no-clean calibrated_default 1.0
run_domain V2_calibrated_friend_half --no-clean calibrated_friend_flat 0.5
run_domain V3_calibrated_friend_full --no-clean calibrated_friend_flat 1.0

"${PYTHON_BIN}" - "${STAGE_DIR}/summary.json" "${CHECKPOINT_PATH}" "${STAGE_DIR}/domains" <<'PY'
import json
import sys
from pathlib import Path

output, checkpoint, domain_dir = sys.argv[1:]
domains = {
    path.stem: json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(Path(domain_dir).glob("V*.json"))
}
if len(domains) != 4:
    raise SystemExit(f"Expected four evaluation domains, got {sorted(domains)}")
Path(output).write_text(json.dumps({
    "status": "completed",
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "domains": domains,
    "holdout_v4_status": "not_preregistered",
}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "evaluation_summary=${STAGE_DIR}/summary.json"
