#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SIM_ROOT="${SIM_ROOT:-${REPO_ROOT}/logs/experiments/go2_v2_contact18_matched25k/sim/20260715_v2_contact18_matched25k_formal}"
EVAL_ROOT="${EVAL_ROOT:-${SIM_ROOT}/matched_policy_evaluation}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-256}"
STEPS="${STEPS:-2400}"
SEED="${SEED:-400}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${EVAL_ROOT}"
CURRENT_STAGE=preflight

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

write_state() {
  printf 'time=%s\nstage=%s\nstatus=%s\ngpu=%s\n' \
    "$(date --iso-8601=seconds)" "$1" "$2" "${CUDA_VISIBLE_DEVICES:-unset}" \
    > "${EVAL_ROOT}/state.txt"
}

on_exit() {
  local code=$?
  if (( code != 0 )); then
    write_state "${CURRENT_STAGE}" failed
  fi
}
trap on_exit EXIT

checkpoint_for() {
  local gap="$1" variant="$2"
  case "${variant}" in
    baseline) cat "${SIM_ROOT}/${gap}/final_policy_rwm_p0/stage/artifact_path.txt" ;;
    trace_r10) cat "${SIM_ROOT}/${gap}/trace_t4_top25_dual/final_policy_trace_r10/stage/artifact_path.txt" ;;
    trace_r25) cat "${SIM_ROOT}/${gap}/trace_t4_top25_dual/final_policy_trace_r25/stage/artifact_path.txt" ;;
    *) echo "Unknown variant: ${variant}" >&2; return 2 ;;
  esac
}

gap_parameters() {
  case "$1" in
    g0) echo "0.0 1.0" ;;
    rr03) echo "0.0 0.3" ;;
    rr05) echo "0.0 0.5" ;;
    p5) echo "5.0 1.0" ;;
    p75) echo "7.5 1.0" ;;
    *) echo "Unknown gap: $1" >&2; return 2 ;;
  esac
}

for gap in g0 rr03 rr05 p5 p75; do
  read -r payload rr_strength < <(gap_parameters "${gap}")
  for variant in baseline trace_r10 trace_r25; do
    output_dir="${EVAL_ROOT}/${gap}/${variant}"
    output_json="${output_dir}/metrics.json"
    if [[ -s "${output_json}" ]]; then
      echo "[reuse] ${gap}/${variant}"
      continue
    fi

    checkpoint="$(checkpoint_for "${gap}" "${variant}")"
    [[ -f "${checkpoint}/actor.pt" ]] || { echo "Missing actor: ${checkpoint}/actor.pt" >&2; exit 2; }
    mkdir -p "${output_dir}"
    CURRENT_STAGE="${gap}_${variant}"
    write_state "${CURRENT_STAGE}" running

    command_args=()
    for command in "${COMMANDS[@]}"; do
      command_args+=("--command_sequence=${command}")
    done

    "${PYTHON_BIN}" scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive_gap.py \
      --checkpoint_path "${checkpoint}" \
      --task Unitree-Go2-Flat-Normal-FixStand-RWM-Pretrain-Ens \
      --device "${DEVICE}" \
      --num_envs "${NUM_ENVS}" \
      --steps "${STEPS}" \
      --seed "${SEED}" \
      --no-clean \
      --randomization_preset calibrated_default \
      --randomization_components friction,mass_com,motor,delay,observation,push,initial_state \
      --randomization_scale 1.0 \
      --payload_mass_kg "${payload}" \
      --rr_calf_strength "${rr_strength}" \
      "${command_args[@]}" \
      --command_switch_steps 300 \
      --output_json "${output_json}" \
      > "${output_dir}/run.log" 2>&1
  done
done

"${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for gap in ("g0", "rr03", "rr05", "p5", "p75"):
    for variant in ("baseline", "trace_r10", "trace_r25"):
        data = json.loads((root / gap / variant / "metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "gap": gap,
            "variant": variant,
            "terminations": int(data["non_timeout_termination_count"]),
            "completed_episodes": int(data["completed_episodes"]),
            "mean_return": float(data["mean_return"]),
            "mean_episode_length": float(data["mean_episode_length"]),
            "error_vel_xy": float(data["error_vel_xy"]),
            "error_vel_yaw": float(data["error_vel_yaw"]),
            "action_abs_mean": float(data["action_abs_mean"]),
            "base_speed_xy": float(data["base_speed_xy"]),
        })

with (root / "matched_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY

write_state done completed
trap - EXIT
echo "evaluation_root=${EVAL_ROOT}"
