"""Independent rollout state on WebSockets sharing one inference engine."""

import numpy as np
import unittest

from openwam.deploy.server import PolicyServer, PolicySession


class FakePolicy:
    def __init__(self):
        self.next_action = 0

    def predict_action(self, obs):
        self.next_action += 1
        return np.array([self.next_action], dtype=np.float32)

    def reset(self):
        self.next_action = 0

    def shutdown(self):
        pass


class IdentityPreprocessor:
    def preprocess(self, obs):
        return obs


class PolicyServerSessionTest(unittest.TestCase):
    def test_reset_and_action_buffer_are_isolated_per_session(self):
        server = PolicyServer(engine=object(), cfg=object(), session_isolation=True)
        server._policy = FakePolicy()
        server._obs_preprocessor = IdentityPreprocessor()
        first = PolicySession(FakePolicy())
        second = PolicySession(FakePolicy())

        with self.assertRaisesRegex(ValueError, "must reset"):
            server.predict({}, session=first)
        first.reset()
        second.reset()
        self.assertEqual(server.predict({}, session=first)["action"], [1.0])
        self.assertEqual(server.predict({}, session=first)["action"], [2.0])
        self.assertEqual(server.predict({}, session=second)["action"], [1.0])
        first.reset()
        self.assertEqual(server.predict({}, session=second)["action"], [2.0])
        self.assertEqual(server.predict({}, session=first)["step"], 1)


if __name__ == "__main__":
    unittest.main()
