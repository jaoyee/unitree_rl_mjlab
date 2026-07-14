"""Export a 45D Go2 FlashSAC expert checkpoint for C++ deployment."""

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


OBS_DIM = 45
ACTION_DIM = 12
DEFAULT_DEPLOY_TEMPLATE = (
    REPO_ROOT
    / "deploy/robots/go2/config/policy/velocity/rwm_proprioceptive_0p5_rr_calf/params/deploy.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_policy_dir", required=True)
    parser.add_argument("--deploy_yaml", default=str(DEFAULT_DEPLOY_TEMPLATE))
    parser.add_argument("--policy_name", required=True)
    parser.add_argument("--training_condition", default="G0")
    parser.add_argument("--dr_condition", required=True)
    parser.add_argument("--simulation_rr_calf_strength", type=float, default=0.5)
    parser.add_argument("--onnx_ir_version", type=int, default=9)
    parser.add_argument("--parity_samples", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    config = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else checkpoint / "flashsac_config.yaml"
    )
    output_dir = Path(args.output_policy_dir).expanduser().resolve()
    deploy_yaml = Path(args.deploy_yaml).expanduser().resolve()
    if not (checkpoint / "actor.pt").is_file():
        raise FileNotFoundError(checkpoint / "actor.pt")
    if not config.is_file():
        raise FileNotFoundError(config)
    if not deploy_yaml.is_file():
        raise FileNotFoundError(deploy_yaml)
    if args.onnx_ir_version != 9:
        raise ValueError("Go2 C++ deployment requires ONNX IR version 9")

    cfg = OmegaConf.load(config)
    OmegaConf.resolve(cfg)
    action_masks = list(cfg.env.get("action_mask_indices", []) or [])
    broken_joints = list(cfg.env.get("broken_joint_names", []) or [])
    if action_masks or broken_joints:
        raise ValueError(
            f"Expert deployment requires full actions and no broken joints: "
            f"action_masks={action_masks}, broken_joints={broken_joints}"
        )

    output_onnx = output_dir / "exported" / "policy.onnx"
    output_yaml = output_dir / "params" / "deploy.yaml"
    parity = export_flashsac_actor_to_onnx(
        checkpoint_path=checkpoint,
        agent_cfg=cfg.agent,
        output_onnx_path=output_onnx,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        onnx_ir_version=args.onnx_ir_version,
        parity_samples=args.parity_samples,
    )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(deploy_yaml, output_yaml)
    signature = read_onnx_signature(output_onnx)
    metadata = {
        "policy_name": args.policy_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint),
        "config_path": str(config),
        "actor_observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "onnx_ir_version": args.onnx_ir_version,
        "onnx_opset": 18,
        "training_condition": args.training_condition,
        "dr_condition": args.dr_condition,
        "training_rr_calf_strength": 1.0,
        "simulation_rr_calf_strength": args.simulation_rr_calf_strength,
        "onnx_signature": {
            "inputs": [asdict(item) for item in signature.inputs],
            "outputs": [asdict(item) for item in signature.outputs],
        },
        "parity": asdict(parity),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"policy_dir={output_dir}")
    print(f"signature={signature}")
    print(f"parity={parity}")


if __name__ == "__main__":
    main()
