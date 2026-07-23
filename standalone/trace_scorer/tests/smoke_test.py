from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_scorer import FORBIDDEN_REWARD_FEATURES, load_scorer


SCORER = ROOT / "checkpoints" / "scorer.pt"


def main() -> None:
    scorer = load_scorer(SCORER, device="cpu")
    assert not (set(scorer.base_feature_names) & FORBIDDEN_REWARD_FEATURES)
    row = {name: 0.0 for name in scorer.base_feature_names}
    scores, missing = scorer.score([row, dict(row)])
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert all(count == 0 for count in missing.values())
    assert scorer.descriptor.checkpoint_sha256 == (
        "61df74670b5cef66c11ad76ab4fb95ccaf65607fe130c9e82abe4b88e16e8e34"
    )
    print("trace_scorer smoke test passed")


if __name__ == "__main__":
    main()
