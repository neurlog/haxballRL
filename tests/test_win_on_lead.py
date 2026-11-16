"""Win reward paid on lead changes (python tests/test_win_on_lead.py).

With win_on_lead the +-1 is paid when a goal changes who is ahead, and over a game it adds
up to exactly the result: +1 win, -1 loss, 0 draw. Here: red scores, red scores, blue scores,
blue scores (2-2, a draw) -> red's rewards per goal are +1, 0, 0, -1 and the total is 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hax.env import HaxEnv  # noqa: E402


def score(env, red_scores: bool) -> np.ndarray:
    """Fire the ball into a goal, then play out the celebration. Returns rewards (red, blue) summed."""
    x = 330.0 if red_scores else -330.0
    env.engine.set_state(np.array([[x, 0.0]]), np.array([[12.0 if red_scores else -12.0, 0.0]]),
                         np.array([[[-200.0, 150.0], [200.0, 150.0]]]), np.zeros((1, 2, 2)))
    env.state = env.engine.snapshot()
    total, before = np.zeros(2), env.score[0].copy()
    for _ in range(120):
        _, r, done, _ = env.step(np.zeros((1, 2), np.int64))
        total += r[0]
        assert not done[0]
    assert (env.score[0] - before).tolist() == ([1, 0] if red_scores else [0, 1]), env.score[0]
    return total


def test_win_on_lead():
    env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=1.0, win_on_lead=True)
    got = [score(env, s) for s in (True, True, False, False)]
    for g, (r, b) in zip(("1-0", "2-0", "2-1", "2-2"), got):
        print(f"  {g}: red {r:+.2f}  blue {b:+.2f}")
    assert np.allclose(got, [[1, -1], [0, 0], [0, 0], [-1, 1]])
    assert abs(sum(r for r, _ in got)) < 1e-6  # a draw sums to 0


if __name__ == "__main__":
    test_win_on_lead()
    print("PASS test_win_on_lead")
