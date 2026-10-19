"""The chunk endpoint must preserve the existing per-step policy contract."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from openwam.deploy.policy import WAMPolicy  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.seeds = []
        self.architecture = type("Architecture", (), {"binary_command_dims": (1,)})()

    def generate(self, conditions):
        seed = conditions["seed"]
        self.seeds.append(seed)
        return {
            "actions": np.asarray(
                [[seed + index / 10, (-1) ** index * 0.75] for index in range(32)],
                dtype=np.float32,
            )
        }


class ActionChunkTest(unittest.TestCase):
    def test_chunk_preserves_actions_and_replan_seeds(self):
        old_engine, new_engine = FakeEngine(), FakeEngine()
        old = WAMPolicy(old_engine, None)
        new = WAMPolicy(new_engine, None)
        old_actions = []
        new_actions = []

        for step in range(70):
            observation = {"seed": 2_000_000_000 + step, "state": [step], "prompt": "same"}
            old_actions.append(old.predict_action(observation))
            if step % 32 == 0:
                new_actions.extend(new.predict_action_chunk(observation))

        np.testing.assert_array_equal(np.stack(old_actions), np.stack(new_actions[:70]))
        self.assertEqual(old_engine.seeds, [2_000_000_000, 2_000_000_032, 2_000_000_064])
        self.assertEqual(new_engine.seeds, old_engine.seeds)
        self.assertTrue(np.isin(np.stack(new_actions)[:, 1], [-1.0, 1.0]).all())


if __name__ == "__main__":
    unittest.main()
