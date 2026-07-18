#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${GAP_ID:?Set GAP_ID to g0, rr05, p5, rr03, or p75}"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU}"
ROOT="${ROOT_OVERRIDE:-${REPO}/logs/rwm_datasets_v5/sim_exact_40_20_20_10_10}"
OUT="${ROOT}/raw/${GAP_ID}.pt"
LOG="${ROOT}/raw/${GAP_ID}.log"
mkdir -p "$(dirname "${OUT}")"

case "${GAP_ID}" in
  g0)   PAYLOAD=0.0; RR=1.0 ;;
  rr05) PAYLOAD=0.0; RR=0.5 ;;
  p5)   PAYLOAD=5.0; RR=1.0 ;;
  rr03) PAYLOAD=0.0; RR=0.3 ;;
  p75)  PAYLOAD=7.5; RR=1.0 ;;
  *) echo "Unsupported GAP_ID=${GAP_ID}" >&2; exit 2 ;;
esac

[[ ! -e "${OUT}" ]] || { echo "Refusing to overwrite ${OUT}" >&2; exit 2; }
export MUJOCO_GL=egl WANDB_MODE=offline
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd "${REPO}"
"${REPO}/.venv/bin/python" \
  scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
  --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
  --device cuda:0 --seed 42 --num_envs 1024 --num_transitions 160000 \
  --save_path "${OUT}" \
  --expert_policy_path "${REPO}/logs/experiments/go2_gap_experts/selected_g0d0/step97656" \
  --collector_mix expert:1.0 --fixed_collector_assignment --collector_assignment_seed 42 \
  --dataset_obs_kind proprioceptive --action_noise_std 0.0 --failure_action_noise_std 0.0 \
  --no-use_domain_randomization --no-use_push_randomization --no-use_observation_noise \
  --payload_mass_range_kg "${PAYLOAD}" "${PAYLOAD}" --rr_calf_strength_range "${RR}" "${RR}" \
  --env_action_noise_std 0.0 --env_action_bias_std 0.0 --env_action_scale_range 1.0 1.0 \
  --env_action_delay_steps 0 0 --chunk_size 0 \
  --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
  --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
  --x_range -0.5 0.5 --signed_x --x_abs_range 0.08 0.50 \
  --y_abs_range 0.05 0.20 --yaw_abs_range 0.08 0.40 \
  --command_resample_interval_min 100 --command_resample_interval_max 250 \
  --save_trace_snapshots --headless 2>&1 | tee "${LOG}"

sha256sum "${OUT}" > "${OUT}.sha256"
