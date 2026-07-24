from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_scorer import FORBIDDEN_REWARD_FEATURES, load_scorer
from trace_scorer.core import REQUIRED_BEHAVIOR_CONTEXT


SCORER = ROOT / "checkpoints" / "scorer.pt"


def main() -> None:
    scorer = load_scorer(SCORER, device="cpu")
    assert not (set(scorer.base_feature_names) & FORBIDDEN_REWARD_FEATURES)
    assert REQUIRED_BEHAVIOR_CONTEXT <= set(scorer.base_feature_names)
    row = {name: 0.0 for name in scorer.base_feature_names}
    scores, missing = scorer.score([row, dict(row)])
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert all(count == 0 for count in missing.values())
    assert scorer.descriptor.checkpoint_sha256 == (
        "61df74670b5cef66c11ad76ab4fb95ccaf65607fe130c9e82abe4b88e16e8e34"
    )
    sparse = {
        name: 0.0
        for name in scorer.base_feature_names
        if name in {
            "survival_fraction",
            "terminal_flag",
            "command_vx_mean",
            "command_vy_mean",
            "command_yaw_mean",
        }
    }
    try:
        scorer.score([sparse])
    except ValueError as error:
        assert "behavior semantics" in str(error) or "missing-feature" in str(error)
    else:
        raise AssertionError("Sparse scorer input did not fail closed.")
    checkpoint = torch.load(SCORER, map_location="cpu", weights_only=False)
    names = list(checkpoint["feature_stats"]["names"])
    feature = "yaw_tracking_error_steady"
    names[names.index(feature)] = "removed_required_semantic"
    names[names.index(f"{feature}_missing")] = (
        "removed_required_semantic_missing"
    )
    checkpoint["feature_stats"]["names"] = names
    with tempfile.TemporaryDirectory() as directory:
        incompatible = Path(directory) / "incompatible.pt"
        torch.save(checkpoint, incompatible)
        try:
            load_scorer(incompatible, device="cpu")
        except ValueError as error:
            assert "omits required behavior semantics" in str(error)
        else:
            raise AssertionError("Incomplete scorer checkpoint did not fail closed.")
    print("trace_scorer smoke test passed")


if __name__ == "__main__":
    main()
