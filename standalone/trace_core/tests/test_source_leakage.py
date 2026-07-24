from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core.source_leakage import audit_source_leakage


class SourceLeakageTest(unittest.TestCase):
    def test_detects_source_marker_in_base_velocity(self) -> None:
        rng = np.random.default_rng(7)
        primary = rng.normal(size=(2000, 48))
        trace = rng.normal(size=(2000, 48))
        trace[:, :3] += 3.0
        report = audit_source_leakage(
            primary,
            trace,
            seed=8,
            maximum_auc=0.70,
            highlighted_indices=(0, 1, 2),
        )
        self.assertFalse(report["passed"])
        self.assertGreater(report["linear_source_auc"], 0.95)
        self.assertEqual(set(report["highlighted_features"]), {"0", "1", "2"})

    def test_matched_sources_pass(self) -> None:
        rng = np.random.default_rng(11)
        shared = rng.normal(size=(4000, 48))
        report = audit_source_leakage(
            shared[:2000],
            shared[2000:],
            seed=9,
            maximum_auc=0.70,
        )
        self.assertTrue(report["passed"])
        self.assertLess(report["linear_source_auc"], 0.60)


if __name__ == "__main__":
    unittest.main()
