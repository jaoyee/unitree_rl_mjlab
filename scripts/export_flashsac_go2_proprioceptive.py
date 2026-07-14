"""Export a 45-dim Go2 proprioceptive FlashSAC actor for C++ deployment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.export import export_flashsac_actor_to_onnx, read_onnx_signature


GO2_ACTOR_OBS_DIM = 45
GO2_ACTION_DIM = 12
DEFAULT_DEPLOY_TEMPLATE = (
    REPO_ROOT
    / "deploy/robots/go2/config/policy/velocity/rwm_proprioceptive_0p5_rr_calf/params/deploy.yaml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_policy_dir", required=True)
    parser.add_argument("--deploy_yaml", default=str(DEFAULT_DEPLOY_TEMPLATE))
    parser.add_argument("--policy_name", default="go2_rwm_proprioceptive_0p5_rr_calf")
    parser.add_argument("--onnx_ir_version", type=int, default=9)
    parser.add_argument("--parity_samples", type=int, default=32)
    parser.add_argument("--parity_atol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else checkpoint_path / "rwm_flashsac_config.yaml"
    )
    output_policy_dir = Path(args.output_policy_dir).expanduser().resolve()
    deploy_template = Path(args.deploy_yaml).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"FlashSAC config not found: {config_path}")
    if not deploy_template.exists():
        raise FileNotFoundError(f"Go2 deploy template not found: {deploy_template}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    output_onnx = output_policy_dir / "exported" / "policy.onnx"
    output_yaml = output_policy_dir / "params" / "deploy.yaml"
    parity = export_flashsac_actor_to_onnx(
        checkpoint_path=checkpoint_path,
        agent_cfg=cfg.agent,
        output_onnx_path=output_onnx,
        obs_dim=GO2_ACTOR_OBS_DIM,
        action_dim=GO2_ACTION_DIM,
        onnx_ir_version=args.onnx_ir_version,
        parity_samples=args.parity_samples,
        parity_atol=args.parity_atol,
    )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(deploy_template, output_yaml)

    metadata = {
        "policy_name": args.policy_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "actor_observation_dim": GO2_ACTOR_OBS_DIM,
        "action_dim": GO2_ACTION_DIM,
        "onnx_ir_version": int(args.onnx_ir_version),
        "observation_order": [
            "base_ang_vel[3]",
            "projected_gravity[3]",
            "velocity_commands[3]",
            "joint_pos_rel[12]",
            "joint_vel_rel[12]",
            "last_action[12]",
        ],
        "policy_joint_order": ["FL", "FR", "RL", "RR"],
        "hardware_joint_order": ["FR", "FL", "RR", "RL"],
        "joint_ids_map": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        "rr_calf_strength_scale": 0.5,
        "onnx_signature": {
            "inputs": [asdict(value) for value in read_onnx_signature(output_onnx).inputs],
            "outputs": [asdict(value) for value in read_onnx_signature(output_onnx).outputs],
        },
        "parity": asdict(parity),
    }
    with (output_policy_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"[Go2-FlashSAC-Export] policy_dir={output_policy_dir}")
    print(f"[Go2-FlashSAC-Export] onnx={output_onnx}")
    print(f"[Go2-FlashSAC-Export] deploy_yaml={output_yaml}")
    print(f"[Go2-FlashSAC-Export] signature={read_onnx_signature(output_onnx)}")
    print(f"[Go2-FlashSAC-Export] parity={parity}")


if __name__ == "__main__":
    main()
