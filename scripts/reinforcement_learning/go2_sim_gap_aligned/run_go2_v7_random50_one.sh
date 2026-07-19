#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${CHECKPOINT_PATH:?}"
: "${MODEL_PATH:?}"
: "${CONDITION:?}"
: "${OUTPUT_DIR:?}"
: "${COMMAND_ROOT:?}"
PY="${REPO}/.venv/bin/python"
SEEDS="${SEEDS:-400 401 402}"
NUM_ENVS="${NUM_ENVS:-256}"
STEPS="${STEPS:-1000}"
SWITCH_STEPS="${SWITCH_STEPS:-50}"

case "${CONDITION}" in
  g0) payload=0.0; rr=1.0 ;;
  rr05) payload=0.0; rr=0.5 ;;
  rr03) payload=0.0; rr=0.3 ;;
  p5) payload=5.0; rr=1.0 ;;
  p75) payload=7.5; rr=1.0 ;;
  *) echo "Unknown condition ${CONDITION}" >&2; exit 2 ;;
esac

cd "${REPO}"
mkdir -p "${OUTPUT_DIR}" "${COMMAND_ROOT}"
export MUJOCO_GL=egl WANDB_MODE=offline

for seed in ${SEEDS}; do
  command_file="${COMMAND_ROOT}/seed_${seed}.json"
  if [[ ! -s "${command_file}" ]]; then
    "${PY}" - "${seed}" "${command_file}.new" <<'PY'
import json, random, sys
seed, output = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
modes = ("stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "xy_yaw")
weights = (0.08, 0.25, 0.10, 0.08, 0.14, 0.17, 0.05, 0.13)
ranges = {"x": (0.05, 0.50), "y": (0.03, 0.20), "yaw": (0.05, 0.40)}
axes = {
    "stand": (), "pure_x": ("x",), "pure_y": ("y",), "pure_yaw": ("yaw",),
    "xy": ("x", "y"), "x_yaw": ("x", "yaw"), "y_yaw": ("y", "yaw"),
    "xy_yaw": ("x", "y", "yaw"),
}
def sample(name, active):
    if not active:
        return 0.0
    lo, hi = ranges[name]
    return round(rng.choice((-1.0, 1.0)) * rng.uniform(lo, hi), 6)
selected = rng.choices(modes, weights=weights, k=20)
commands = [[sample("x", "x" in axes[m]), sample("y", "y" in axes[m]),
             sample("yaw", "yaw" in axes[m])] for m in selected]
json.dump({"seed": seed, "modes": selected, "commands": commands}, open(output, "w"), indent=2)
PY
    mv "${command_file}.new" "${command_file}"
  fi
  output_json="${OUTPUT_DIR}/seed_${seed}.json"
  [[ -s "${output_json}" ]] && continue
  mapfile -t commands < <("${PY}" - "${command_file}" <<'PY'
import json, sys
for command in json.load(open(sys.argv[1]))["commands"]:
    print(",".join(str(value) for value in command))
PY
  )
  command_args=()
  for command in "${commands[@]}"; do command_args+=("--command_sequence=${command}"); done
  "${PY}" scripts/reinforcement_learning/rwm_flashsac/eval_flashsac_go2_mjlab_proprioceptive_gap.py \
    --checkpoint_path "${CHECKPOINT_PATH}" --model_path "${MODEL_PATH}" \
    --task Unitree-Go2-Flat-Normal-FixStand-RWM-Pretrain-Ens \
    --device cuda:0 --num_envs "${NUM_ENVS}" --steps "${STEPS}" --seed "${seed}" \
    --command_switch_steps "${SWITCH_STEPS}" --payload_mass_kg "${payload}" \
    --rr_calf_strength "${rr}" --clean "${command_args[@]}" \
    --output_json "${output_json}.new" > "${OUTPUT_DIR}/seed_${seed}.log" 2>&1
  mv "${output_json}.new" "${output_json}"
done
