"""Reward-hacking checks for the 3v3 team-play rewards in hax/env.py.

    python tests/test_team_rewards.py        (or: pytest tests/)

Players are placed by hand (red 0-2, blue 3-5) and driven with fixed inputs, then we check
which situations the pass / crowding / chase rewards pay for.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hax.env import HaxEnv, keys_to_action  # noqa: E402

FAR_BLUE = [(-300, 150), (-300, -150), (300, 150)]


def make_env(pass_reward=0.0, crowd=0.0, speed=0.0, spirit=0.0):
    env = HaxEnv(1, team_size=3, ping_ms=0.0, jitter_ms=0.0, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0)
    env.set_reward_weights(speed=speed)
    env.set_team_rewards(pass_reward=pass_reward, crowd_penalty=crowd, team_spirit=spirit)
    return env


def place(env, ball, red, blue=FAR_BLUE):
    pos = np.array([list(red) + list(blue)], float)
    env.engine.set_state(np.array([ball], float), np.zeros((1, 2)), pos, np.zeros((1, 6, 2)))
    env.state = env.engine.snapshot()


def play(env, keys_fn, decisions, touch_log=None):
    """keys_fn(t) -> list of 6 (dx, dy, kick) in WORLD coordinates. Returns summed (6,) reward.
    If touch_log is a list, the sequence of players who touched the ball is appended to it."""
    total = np.zeros(6)
    for t in range(decisions):
        keys = keys_fn(t)
        a = [keys_to_action(dx * (1 if k < 3 else -1), dy, kick) for k, (dx, dy, kick) in enumerate(keys)]
        _, r, _, _ = env.step(np.array([a]))
        total += r[0]
        if touch_log is not None and env.last_toucher[0] >= 0 and (not touch_log or touch_log[-1] != env.last_toucher[0]):
            touch_log.append(int(env.last_toucher[0]))
    return total


STILL = (0, 0, False)


def test_forward_pass_pays_and_is_zero_sum():
    env = make_env(pass_reward=1.0)
    place(env, (-172, 0), [(-200, 0), (150, 0), (-200, 150)])  # red 0 kicks right to red 1
    r = play(env, lambda t: [(1, 0, t < 3)] + [STILL] * 5, 90)
    print(f"  300+ px forward pass: red {r[:3].round(2)}, blue {r[3:].round(2)}, passes {env.passes[0]}")
    assert env.passes[0, 0] == 1 and r[0] > 0.9 and np.allclose(r[:3], r[0]), "whole red team gets the pass reward"
    assert np.allclose(r[3:], -r[0]), "zero-sum: blue loses what red gains"


def test_backward_pass_earns_nothing():
    env = make_env(pass_reward=1.0)
    place(env, (122, 0), [(150, 0), (-200, 0), (-200, 150)])  # red 0 kicks left (towards own goal) to red 1
    r = play(env, lambda t: [(-1, 0, t < 3)] + [STILL] * 5, 90)
    print(f"  backward pass:        red {r[:3].round(2)}, passes {env.passes[0]}")
    assert env.passes[0].sum() == 0 and np.allclose(r, 0)


def test_short_pass_earns_nothing():
    env = make_env(pass_reward=1.0)
    place(env, (28, 0), [(0, 0), (85, 0), (-200, 150)])  # red 1 only ~30 px away
    r = play(env, lambda t: [(1, 0, t < 3)] + [STILL] * 5, 60)
    print(f"  short pass (<100 px): red {r[:3].round(2)}, passes {env.passes[0]}")
    assert env.passes[0].sum() == 0 and np.allclose(r, 0)


def test_intercepted_pass_earns_nothing():
    env = make_env(pass_reward=1.0)
    place(env, (-172, 0), [(-200, 0), (200, 0), (-200, 150)], blue=[(0, 0), (-300, -150), (300, 150)])
    log = []
    r = play(env, lambda t: [(1, 0, t < 3)] + [STILL] * 5, 120, log)  # blue 3 stands in the passing lane
    print(f"  intercepted pass:     red {r[:3].round(2)}, passes {env.passes[0]}, touch order {log}")
    assert log[:2] == [0, 3], "setup: blue 3 should touch the ball right after red 0"
    assert env.passes[0, 0] == 0 and r[0] <= 0.0


def test_corner_ping_pong_earns_nothing():
    env = make_env(pass_reward=1.0)
    # two red players face each other against the top wall near the corner, ball squeezed between
    # them, and take turns kicking it into each other for 10 s (dozens of teammate-to-teammate touches)
    place(env, (320, -158), [(288, -158), (352, -158), (-200, 150)])
    log = []
    r = play(env, lambda t: [(1, 0, t % 2 == 0), (-1, 0, t % 2 == 1), STILL, STILL, STILL, STILL], 300, log)
    switches = sum(1 for a, b in zip(log, log[1:]) if {a, b} == {0, 1})
    print(f"  corner ping-pong 10s: red {r[:3].round(2)}, passes {env.passes[0]}, "
          f"{switches} touches passed between red 0 and red 1")
    assert switches >= 10, "setup: the two red players should keep touching the ball in turn"
    assert env.passes[0].sum() == 0 and abs(r[0]) < 1e-9


def test_crowding_penalty():
    env = make_env(crowd=1.0)
    place(env, (0, 100), [(-100, 0), (-100, 30), (-100, -30)])  # three red players bunched up
    bunched = play(env, lambda t: [STILL] * 6, 30)
    env = make_env(crowd=1.0)
    place(env, (0, 100), [(-100, 0), (-250, 120), (100, -120)])  # spread out
    spread = play(env, lambda t: [STILL] * 6, 30)
    print(f"  crowding 1 s: bunched red {bunched[:3].round(2)}, spread red {spread[:3].round(2)}")
    assert (bunched[:3] < -1.0).all() and np.allclose(spread, 0)
    assert np.allclose(bunched[3:], 0), "blue isn't bunched"


def test_only_nearest_player_is_paid_to_chase():
    env = make_env(speed=1.0)
    place(env, (0, 0), [(-100, 0), (-200, 0), (-300, 0)], blue=[(300, 150), (300, -150), (350, 0)])
    r = play(env, lambda t: [(1, 0, False)] * 3 + [STILL] * 3, 20)  # all three red run at the ball
    print(f"  three red chase the ball: red {r[:3].round(2)}")
    assert r[0] > 1.0 and r[1] == 0.0 and r[2] == 0.0


def test_team_spirit_shares_individual_rewards():
    env = make_env(speed=1.0, spirit=1.0)
    place(env, (0, 0), [(-100, 0), (-200, 0), (-300, 0)], blue=[(300, 150), (300, -150), (350, 0)])
    r = play(env, lambda t: [(1, 0, False)] * 3 + [STILL] * 3, 20)
    print(f"  same, team spirit 1.0:    red {r[:3].round(2)}")
    assert r[0] > 0.3 and np.allclose(r[:3], r[0])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
