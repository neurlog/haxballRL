"""Corner-stall penalty (python tests/test_corner_box.py).

Run v5 pinned the ball in the corners (26% of play at 53k games): with only zero-sum rewards a
pin cost nobody anything. Now a ball that sits in a corner square for more than `corner_hold_s`
costs EVERY player `corner_penalty` per decision until it leaves; before that, and anywhere else
on the pitch, it costs nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hax.env import HaxEnv  # noqa: E402


def hold(ball, seconds, box=40.0):
    env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0,
                 corner_box=box, corner_hold_s=5.0, corner_penalty=0.05)
    env.engine.set_state(np.array([ball], float), np.zeros((1, 2)),
                         np.array([[[-200.0, 0.0], [200.0, 0.0]]]), np.zeros((1, 2, 2)))
    env.state = env.engine.snapshot()
    rewards, hot = [], []
    for _ in range(int(seconds * 30)):  # 30 decisions per second
        _, r, _, _ = env.step(np.zeros((1, 2), np.int64))
        rewards.append(r[0].copy())
        hot.append(env.frame(0)["corner_hot"])
    return np.array(rewards), np.array(hot)


def test_free_for_5s_then_both_pay():
    r, hot = hold((355.0, -155.0), 8.0)  # ball resting in blue's top corner
    first5, after = r[:149], r[152:]
    print(f"  first 5 s: {first5.sum(0)} | after: {after.mean(0)} per decision, {after.sum(0)} over 3 s")
    assert np.abs(first5).max() == 0.0
    assert np.allclose(after, -0.05, atol=1e-6)  # both players, the full penalty every decision
    assert not hot[:149].any() and hot[152:].all()


def test_every_corner():
    for ball in ((355.0, 155.0), (-355.0, 155.0), (-355.0, -155.0)):
        r, _ = hold(ball, 6.0)
        assert np.allclose(r[-1], -0.05, atol=1e-6), ball


def test_outside_the_box_is_free():
    for ball in ((300.0, -155.0), (355.0, -100.0), (0.0, 150.0)):  # by a wall but not in a corner square
        r, hot = hold(ball, 7.0)
        assert np.abs(r).max() == 0.0 and not hot.any(), ball


def test_scale_zero_measures_but_charges_nothing():
    env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0,
                 corner_box=40.0, corner_hold_s=5.0, corner_penalty=0.05)
    env.set_corner_scale(0.0)  # training before --corner-start-games
    env.engine.set_state(np.array([(355.0, -155.0)]), np.zeros((1, 2)),
                         np.array([[[-200.0, 0.0], [200.0, 0.0]]]), np.zeros((1, 2, 2)))
    env.state = env.engine.snapshot()
    for _ in range(8 * 30):
        _, r, _, _ = env.step(np.zeros((1, 2), np.int64))
        assert np.abs(r).max() == 0.0
    assert env.corner_pen_ticks[0] > 0  # still logged


def test_off_by_default():
    env = HaxEnv(1, team_size=1)
    assert env.corner_box == 0.0 and env.frame(0)["corner_box"] == 0.0


if __name__ == "__main__":
    for t in (test_free_for_5s_then_both_pay, test_every_corner, test_outside_the_box_is_free,
              test_scale_zero_measures_but_charges_nothing, test_off_by_default):
        t()
        print("PASS", t.__name__)
