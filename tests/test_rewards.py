"""Reward-hacking checks for the dense rewards in hax/env.py.

    python tests/test_rewards.py        (or: pytest tests/)

The touch reward was once farmed by pinning the ball in a corner and spamming kick: every
kick bounced the ball off the wall back into the player. These tests replay that exploit
in all four corners and check it now earns about nothing, while a real shot still pays.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hax.env import HaxEnv, keys_to_action  # noqa: E402

SECONDS = 10
DECISIONS = SECONDS * 30


def make_env(touch=1.0, speed=0.0, btg=0.0):
    env = HaxEnv(1, team_size=1, ping_ms=0.0, jitter_ms=0.0, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0)
    env.set_reward_weights(speed=speed, touch=touch, ball_to_goal=btg)
    return env


def place(env, ball, red, blue=(0.0, 150.0)):
    env.engine.set_state(np.array([ball], float), np.zeros((1, 2)), np.array([[red, blue]], float),
                         np.zeros((1, 2, 2)))
    env.state = env.engine.snapshot()


def play(env, red_keys, decisions=DECISIONS):
    """red_keys(t) -> (dx, dy, kick) in WORLD coordinates (red's frame = world frame).
    Returns (total red reward, number of kicks that connected)."""
    total, kicks = 0.0, 0
    prev_speed = np.linalg.norm(env.state.ball_vel[0])
    for t in range(decisions):
        dx, dy, k = red_keys(t)
        a = np.array([[keys_to_action(dx, dy, k), 0]])
        _, r, _, _ = env.step(a)
        total += float(r[0, 0])
        speed = np.linalg.norm(env.state.ball_vel[0])
        kicks += speed > prev_speed + 2.0
        prev_speed = speed
    return total, kicks


CORNERS = {  # ball spot, player spot, direction pushing into the corner (y is DOWN in HaxBall)
    "top-right (attacking corner)": ((356, -156), (330, -130), (1, -1)),
    "bottom-right (attacking corner)": ((356, 156), (330, 130), (1, 1)),
    "top-left (own corner)": ((-356, -156), (-330, -130), (-1, -1)),
    "bottom-left (own corner)": ((-356, 156), (-330, 130), (-1, 1)),
}


def test_corner_kick_spam_earns_nothing():
    for name, (ball, red, (dx, dy)) in CORNERS.items():
        env = make_env(touch=1.0)
        place(env, ball, red)
        total, kicks = play(env, lambda t, dx=dx, dy=dy: (dx, dy, t % 2 == 0))  # tap kick every other decision
        print(f"  corner spam, {name:32s}: {kicks:3d} kicks in {SECONDS}s -> touch reward {total:+.2f}")
        assert kicks >= 20, f"{name}: the exploit didn't actually kick ({kicks} kicks); test setup is wrong"
        # must not pay: the old touch reward earned about +1 per kick here (34 per 10 s at weight 1).
        # Negative is fine - kicking the ball into a wall loses goal-ward speed on the rebound.
        assert total < 1.0, f"{name}: corner kick-spam earned {total:+.2f} touch reward in {SECONDS}s"


def test_shot_at_goal_pays():
    env = make_env(touch=1.0)
    place(env, (250, 0), (222, 0))
    total, kicks = play(env, lambda t: (1, 0, t < 3), decisions=30)
    print(f"  one shot straight at goal: {kicks} kick(s) -> touch reward {total:+.2f}")
    assert kicks >= 1 and total > 0.5


def test_shot_at_own_goal_costs():
    env = make_env(touch=1.0)
    place(env, (-250, 0), (-222, 0))
    total, kicks = play(env, lambda t: (-1, 0, t < 3), decisions=30)
    print(f"  one shot at own goal:      {kicks} kick(s) -> touch reward {total:+.2f}")
    assert kicks >= 1 and total < -0.5


def test_back_and_forth_next_to_the_ball_earns_nothing():
    """The speed reward is signed, so it only pays for the distance actually closed:
    total ~= (start distance - end distance) / (2 ticks x 2.4 px/tick), however long you dance."""
    for seconds in (10, 30):
        env = make_env(touch=0.0, speed=1.0)
        place(env, (0, 100), (-200, 100))  # ball in play, player 200 px away
        d0 = 200.0
        total, _ = play(env, lambda t: (1 if (t // 10) % 2 == 0 else -1, 0, False), decisions=seconds * 30)
        d1 = float(np.linalg.norm(env.state.ball_pos[0] - env.state.player_pos[0, 0]))
        honest = (d0 - d1) / (2 * 2.4)
        print(f"  back and forth near the ball for {seconds}s: speed reward {total:+.2f}, "
              f"distance closed {d0 - d1:+.0f} px (worth {honest:+.2f})")
        assert abs(total - honest) < 1.5, "speed reward paid more than the distance actually closed"
        assert total < 200 / 4.8 + 1.0, "can never earn more than closing the whole gap once"


def _kickoff_env(penalty=0.002, grace=5.0):
    env = HaxEnv(1, team_size=1, ping_ms=0.0, jitter_ms=0.0, kickoff_timeout_s=0.0, goal_reward=0.0,
                 win_reward=0.0, kickoff_penalty=penalty, kickoff_grace_s=grace)
    env.reset()  # fresh game: red kicks off
    return env


def _run(env, decisions, red_keys):
    red = blue = 0.0
    for t in range(decisions):
        _, r, _, _ = env.step(np.array([[keys_to_action(*red_keys(t)), 0]]))
        red, blue = red + float(r[0, 0]), blue + float(r[0, 1])
    return red, blue


def test_kickoff_penalty_only_after_grace():
    red, blue = _run(_kickoff_env(), int(4.9 * 30), lambda t: (0, 0, False))
    print(f"  red waits 4.9 s at its kickoff: red {red:+.3f}, blue {blue:+.3f}")
    assert red == 0.0 and blue == 0.0, "no penalty inside the 5 s grace period"
    red, blue = _run(_kickoff_env(), 10 * 30, lambda t: (0, 0, False))
    print(f"  red waits 10 s at its kickoff:  red {red:+.3f}, blue {blue:+.3f}  (expected -0.300)")
    assert abs(red - (-0.002 * 5 * 30)) < 0.01, "5 s past the grace period should cost 150 decisions x 0.002"
    assert blue == 0.0, "the defending team is never penalised"


def test_kickoff_penalty_stops_once_taken():
    env = _kickoff_env()
    red, _ = _run(env, 6 * 30, lambda t: (0, 0, False))  # dawdle 6 s (1 s past the grace period)
    t = 0
    while env.kickoff_waiting()[0]:  # then walk into the ball (about 3 s from spawn)
        r, _ = _run(env, 1, lambda _: (1, 0, True))
        red += r
        t += 1
        assert t < 300, "red never reached the ball"
    expected = -0.002 * (6 * 30 + t - 5 * 30)
    after, _ = _run(env, 30, lambda _: (0, 0, False))  # ball in play: no more penalty
    print(f"  red waits 6 s, walks {t / 30:.1f} s to take it: red {red:+.3f} (expected {expected:+.3f}), "
          f"next 1 s {after:+.3f}")
    assert abs(red - expected) < 0.01 and after == 0.0


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
