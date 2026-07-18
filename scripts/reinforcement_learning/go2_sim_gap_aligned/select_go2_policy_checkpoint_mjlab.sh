#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${POLICY_OUTPUT_DIR:?Set POLICY_OUTPUT_DIR to the directory containing step checkpoints}"
: "${GAP_ID:?Set GAP_ID to g0, rr03, rr05, p5, or p75}"
: "${VALIDATION_ROOT:?Set VALIDATION_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VALIDATION_STRIDE="${VALIDATION_STRIDE:-5000}"
VALIDATION_NUM_ENVS="${VALIDATION_NUM_ENVS:-128}"
VALIDATION_STEPS="${VALIDATION_STEPS:-2400}"
VALIDATION_SEED="${VALIDATION_SEED:-401}"
VALIDATION_MAX_ERROR_XY="${VALIDATION_MAX_ERROR_XY:-0.15}"
VALIDATION_MAX_ERROR_YAW="${VALIDATION_MAX_ERROR_YAW:-0.15}"
VALIDATION_MIN_EPISODE_LENGTH="${VALIDATION_MIN_EPISODE_LENGTH:-300}"

case "${GAP_ID}" in
  g0) PAYLOAD_KG=0.0; RR_STRENGTH=1.0 ;;
  rr03) PAYLOAD_KG=0.0; RR_STRENGTH=0.3 ;;
  rr05) PAYLOAD_KG=0.0; RR_STRENGTH=0.5 ;;
  p5) PAYLOAD_KG=5.0; RR_STRENGTH=1.0 ;;
  p75) PAYLOAD_KG=7.5; RR_STRENGTH=1.0 ;;
  *) echo "Unsupported GAP_ID=${GAP_ID}" >&2; exit 2 ;;
esac

mkdir -p "${VALIDATION_ROOT}"
mapfile -t CHECKPOINTS < <(
  find "${POLICY_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'step*' -print |
    while read -r checkpoint; do
      step="${checkpoint##*/step}"
      if [[ "${step}" =~ ^[0-9]+$ ]] && (( step % VALIDATION_STRIDE == 0 )); then
        printf '%012d %s\n' "${step}" "${checkpoint}"
      fi
    done | sort -n
)
LATEST="$(find "${POLICY_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'step*' -print |
  sort -V | tail -1)"
if [[ -n "${LATEST}" ]] && ! printf '%s\n' "${CHECKPOINTS[@]}" | grep -Fq " ${LATEST}"; then
  step="${LATEST##*/step}"
  CHECKPOINTS+=("$(printf '%012d %s' "${step}" "${LATEST}")")
fi
if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "No checkpoints found under ${POLICY_OUTPUT_DIR}" >&2
  exit 3
fi

for row in "${CHECKPOINTS[@]}"; do
  checkpoint="${row#* }"
  step="${checkpoint##*/step}"
  output="${VALIDATION_ROOT}/step${step}/metrics.json"
  [[ -s "${output}" ]] && continue
  mkdir -p "$(dirname "${output}")"
  "${PYTHON_BIN}" \
    scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive_gap.py \
    --checkpoint_path "${checkpoint}" \
    --task Unitree-Go2-Flat-Normal-FixStand-RWM-Pretrain-Ens \
    --device cuda:0 \
    --num_envs "${VALIDATION_NUM_ENVS}" \
    --steps "${VALIDATION_STEPS}" \
    --seed "${VALIDATION_SEED}" \
    --no-clean \
    --randomization_preset calibrated_default \
    --randomization_components friction,mass_com,motor,delay,observation,push,initial_state \
    --randomization_scale 1.0 \
    --payload_mass_kg "${PAYLOAD_KG}" \
    --rr_calf_strength "${RR_STRENGTH}" \
    --command_sequence=0.0,0.0,0.0 \
    --command_sequence=0.5,0.0,0.0 \
    --command_sequence=-0.4,0.0,0.0 \
    --command_sequence=0.0,0.2,0.0 \
    --command_sequence=0.0,-0.2,0.0 \
    --command_sequence=0.0,0.0,0.4 \
    --command_sequence=0.0,0.0,-0.4 \
    --command_sequence=0.4,0.2,0.3 \
    --command_switch_steps 300 \
    --output_json "${output}"
done

"${PYTHON_BIN}" - "${VALIDATION_ROOT}" "${VALIDATION_ROOT}/selection.json" \
  "${VALIDATION_MAX_ERROR_XY}" "${VALIDATION_MAX_ERROR_YAW}" \
  "${VALIDATION_MIN_EPISODE_LENGTH}" <<'PY'
import glob
import json
import sys
from pathlib import Path

root, output, max_error_xy, max_error_yaw, min_episode_length = sys.argv[1:]
max_error_xy = float(max_error_xy)
max_error_yaw = float(max_error_yaw)
min_episode_length = float(min_episode_length)
rows = []
for path in glob.glob(f"{root}/step*/metrics.json"):
    metrics = json.loads(Path(path).read_text())
    checkpoint = str(metrics["checkpoint_path"])
    rows.append({
        "checkpoint_path": checkpoint,
        "step": int(Path(checkpoint).name.removeprefix("step")),
        "mean_episode_length": float(metrics["mean_episode_length"]),
        "mean_return": float(metrics["mean_return"]),
        "terminated_count": int(metrics["terminated_count"]),
        "error_vel_xy": float(metrics["error_vel_xy"]),
        "error_vel_yaw": float(metrics["error_vel_yaw"]),
        "metrics_path": str(Path(path).resolve()),
    })
if not rows:
    raise SystemExit("No completed validation metrics.")
eligible = [
    row
    for row in rows
    if row["error_vel_xy"] <= max_error_xy
    and row["error_vel_yaw"] <= max_error_yaw
    and row["mean_episode_length"] >= min_episode_length
]
if not eligible:
    failure = {
        "selection_protocol": "mjlab_calibrated_default_tracking_gate_survival_first_v2",
        "status": "failed_quality_gate",
        "quality_gate": {
            "max_error_vel_xy": max_error_xy,
            "max_error_vel_yaw": max_error_yaw,
            "min_mean_episode_length": min_episode_length,
        },
        "all_checkpoints": sorted(rows, key=lambda row: row["step"]),
    }
    Path(output).write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
    raise SystemExit(
        "No checkpoint passed the MJLab tracking/stability gate: "
        f"error_xy<={max_error_xy}, error_yaw<={max_error_yaw}, "
        f"episode_length>={min_episode_length}."
    )
# Survival is the primary safety criterion. Return and tracking errors break close ties.
best = max(
    eligible,
    key=lambda row: (
        row["mean_episode_length"],
        -row["terminated_count"],
        row["mean_return"],
        -(row["error_vel_xy"] + row["error_vel_yaw"]),
    ),
)
payload = {
    "selection_protocol": "mjlab_calibrated_default_tracking_gate_survival_first_v2",
    "status": "passed",
    "quality_gate": {
        "max_error_vel_xy": max_error_xy,
        "max_error_vel_yaw": max_error_yaw,
        "min_mean_episode_length": min_episode_length,
    },
    "best_checkpoint": best["checkpoint_path"],
    "best": best,
    "all_checkpoints": sorted(rows, key=lambda row: row["step"]),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(best["checkpoint_path"])
PY
