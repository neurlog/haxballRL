"""Ball-to-goal corner exploit (python tests/test_btg_zero_sum.py).

Run v4 camped the corners: the goal direction points at the nearest point of the goal mouth,
so in a corner BOTH goals lie partly towards the middle and a ball rolling out of a corner
paid both players. The zero-sum version, (yours - the opponent's) / 2, must sum to 0, and a
ball rolled at the opponent's goal must still pay the attacker.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hax.env import HaxEnv  # noqa: E402


def roll(zero_sum: bool, ball, vel, decisions=20):
    env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0, btg_zero_sum=zero_sum)
    env.set_reward_weights(ball_to_goal=1.0)
    env.engine.set_state(np.array([ball], float), np.array([vel], float),
                         np.array([[[-200.0, 150.0], [200.0, 150.0]]]), np.zeros((1, 2, 2)))
    env.state = env.engine.snapshot()
    total = np.zeros(2)
    for _ in range(decisions):
        _, r, _, _ = env.step(np.zeros((1, 2), np.int64))
        total += r[0]
    return total


def test_corner_exit_pays_nobody_for_free():
    for corner, ball in (("red's attacking corner", (345.0, -150.0)), ("red's own corner", (-345.0, 150.0))):
        vel = (0.0, 3.0) if ball[1] < 0 else (0.0, -3.0)  # rolling out of the corner along the end line
        plain, zs = roll(False, ball, vel), roll(True, ball, vel)
        print(f"  ball leaves {corner}: plain red {plain[0]:+.2f} blue {plain[1]:+.2f} | "
              f"zero-sum red {zs[0]:+.2f} blue {zs[1]:+.2f}")
        assert plain.min() > 0.05  # the old version paid both players (the exploit)
        assert abs(zs.sum()) < 1e-5                # zero-sum: nothing for free


def test_attack_still_pays():
    zs = roll(True, (0.0, 0.0), (4.0, 0.0))  # ball rolled at blue's goal from the centre
    print(f"  ball rolled at blue's goal: red {zs[0]:+.2f} blue {zs[1]:+.2f}")
    assert zs[0] > 1.0 and abs(zs.sum()) < 1e-5


if __name__ == "__main__":
    test_corner_exit_pays_nobody_for_free()
    print("PASS test_corner_exit_pays_nobody_for_free")
    test_attack_still_pays()
    print("PASS test_attack_still_pays")
