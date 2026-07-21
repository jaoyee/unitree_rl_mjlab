import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "flash_rl/agents/flashSAC/update_schedule.py"
SPEC = importlib.util.spec_from_file_location("flashsac_update_schedule", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
should_update_actor = MODULE.should_update_actor


class TestFlashSACActorWarmup(unittest.TestCase):
    def test_actor_is_frozen_before_learning_start(self) -> None:
        self.assertFalse(should_update_actor(0, 3000, 2))
        self.assertFalse(should_update_actor(2999, 3000, 2))

    def test_actor_period_is_relative_to_learning_start(self) -> None:
        self.assertTrue(should_update_actor(3000, 3000, 2))
        self.assertFalse(should_update_actor(3001, 3000, 2))
        self.assertTrue(should_update_actor(3002, 3000, 2))

    def test_default_preserves_existing_schedule(self) -> None:
        self.assertTrue(should_update_actor(0, 0, 2))
        self.assertFalse(should_update_actor(1, 0, 2))
        self.assertTrue(should_update_actor(2, 0, 2))

    def test_invalid_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            should_update_actor(0, -1, 2)
        with self.assertRaises(ValueError):
            should_update_actor(0, 0, 0)


if __name__ == "__main__":
    unittest.main()
