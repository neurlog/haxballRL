"""Idle-ball penalty (python tests/test_idle_ball.py).

Run v6 parked both bots in their own goals and left the ball alone (98% of games 0-0). Now, once
the ball has stayed within `idle_dist` px for more than `idle_hold_s` of play, the player NEAREST
the ball loses `idle_penalty` per decision until the ball travels `idle_dist` px.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hax.env import HaxEnv, keys_to_action  # noqa: E402


def setup(ball, vel=(0.0, 0.0), players=((-300.0, 0.0), (300.0, 0.0))):
    env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0,
                 idle_dist=50.0, idle_hold_s=5.0, idle_penalty=0.1)
    env.engine.set_state(np.array([ball], float), np.array([vel], float), np.array([players], float), np.zeros((1, 2, 2)))
    env.state = env.engine.snapshot()
    env.idle_anchor[:] = ball
    return env


def run(env, seconds, actions=(0, 0)):
    out = []
    for _ in range(int(seconds * 30)):
        _, r, _, _ = env.step(np.array([actions], np.int64))
        out.append(r[0].copy())
    return np.array(out)


def test_nearest_player_pays_after_5s():
    env = setup((100.0, 0.0))  # still ball, blue (at x=300) is nearer than red (at x=-300)
    r = run(env, 8.0)
    print(f"  first 5 s: {r[:149].sum(0)} | after: {r[152:].mean(0)} per decision")
    assert np.abs(r[:149]).max() == 0.0
    assert np.allclose(r[152:, 1], -0.1, atol=1e-6) and np.abs(r[152:, 0]).max() == 0.0  # blue only
    assert env.frame(0)["idle_hot"]


def test_rolling_ball_is_free():
    env = setup((-300.0, 100.0), vel=(3.0, 0.0))  # ball rolling across the pitch, travels > 50 px
    r = run(env, 4.0)
    assert np.abs(r).max() == 0.0


def test_moving_the_ball_resets_the_clock():
    env = setup((0.0, -120.0), players=((-40.0, -120.0), (300.0, 100.0)))
    run(env, 6.0)  # idle: red (40 px away) is paying
    assert env.idle_ticks[0] > env.idle_hold
    r = run(env, 2.0, actions=(3 + 9, 0))  # red runs right and kicks the ball away
    assert env.idle_ticks[0] < env.idle_hold and r[-1, 0] == 0.0


def test_touch_in_corner_box_resets():
    # ball held against the top-right corner by red: inside a corner square every touch resets the
    # idle clock (the corner rule judges that instead); the same hold on an open wall does not
    for ball, player, in_box in (((355.0, -155.0), (330.0, -140.0), True), ((0.0, -155.0), (0.0, -130.0), False)):
        env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0,
                     idle_dist=40.0, idle_hold_s=5.0, idle_penalty=0.1, corner_box=50.0)
        env.engine.set_state(np.array([ball]), np.zeros((1, 2)), np.array([[player, (300.0, 100.0)]]), np.zeros((1, 2, 2)))
        env.state = env.engine.snapshot()
        env.idle_anchor[:] = ball
        paid, touching = 0.0, []
        for _ in range(7 * 30):  # red keeps steering into the ball, so it stays in contact
            d = env.state.ball_pos[0] - env.state.player_pos[0, 0]
            a = keys_to_action(int(np.sign(np.round(d[0]))), int(np.sign(np.round(d[1]))), False)
            _, r, _, _ = env.step(np.array([[a, 0]], np.int64))
            paid -= r[0, 0]
            touching.append(np.linalg.norm(d) < 26.5)
        print(f"  {'corner' if in_box else 'open wall'}: touching {np.mean(touching):.0%}, idle penalty paid {paid:.2f}")
        assert np.mean(touching) > 0.9
        assert (paid == 0.0) == in_box


def test_scale_zero_measures_but_charges_nothing():
    env = setup((100.0, 0.0))
    env.set_idle_scale(0.0)  # training before --idle-start-games
    r = run(env, 8.0)
    assert np.abs(r).max() == 0.0 and env.idle_pen_ticks[0] > 0  # still logged
    env.set_idle_scale(0.5)  # halfway through the ramp
    r = run(env, 1.0)
    assert np.allclose(r[:, 1], -0.05, atol=1e-6)


def test_1v1_schedule():
    import importlib.util
    from hax.train import build_parser, ramp
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("t1", os.path.join(root, "1v1-classic", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = build_parser(mod.DEFAULTS).parse_args([])
    for start, length in ((a.idle_start_games, a.idle_ramp_games), (a.corner_start_games, a.corner_ramp_games)):
        got = tuple(ramp(g, start, length) for g in (0, 4999, 5500, 6000, 50000))
        assert got == (0.0, 0.0, 0.5, 1.0, 1.0), got


def test_share_by_score_and_distance():
    # user's table: winning (nearest 1/2, others 0), level (1, 1/2), losing (1, 1)
    table = ((0.5, 0.0), (1.0, 0.5), (1.0, 1.0))
    # ball still at x=100: blue (at 300) is nearest, red (at -300) is far away
    for score, want in (((1, 0), (0.0, 0.1)),    # red winning & far: 0 | blue losing & nearest: full
                        ((0, 0), (0.05, 0.1)),   # level: far pays half, nearest pays full
                        ((0, 1), (0.1, 0.05))):  # red losing & far: full | blue winning & nearest: half
        env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0,
                     idle_dist=40.0, idle_hold_s=5.0, idle_penalty=0.1, idle_table=table)
        env.engine.set_state(np.array([(100.0, 0.0)]), np.zeros((1, 2)),
                             np.array([[(-300.0, 0.0), (300.0, 0.0)]]), np.zeros((1, 2, 2)))
        env.state = env.engine.snapshot()
        env.idle_anchor[:] = (100.0, 0.0)
        env.score[0] = score
        r = run(env, 6.0)
        print(f"  score {score[0]}-{score[1]}: red (far) {-r[-1, 0]:.3f}, blue (nearest) {-r[-1, 1]:.3f} per decision")
        assert np.allclose(-r[-1], want, atol=1e-6), (score, -r[-1], want)


def test_preset_winner_never_pays():
    # hax/presets.py: winning pays nothing, level (nearest 1, others 1/2), losing (1, 1)
    from hax.presets import REWARDS
    from hax.train import idle_table
    got = np.asarray(idle_table(REWARDS["idle_table"]), float).reshape(3, 2)
    assert np.allclose(got, ((0.0, 0.0), (1.0, 0.5), (1.0, 1.0))), got
    # blue (nearest) leads 0-1: only red, losing and far away, pays
    env = HaxEnv(1, team_size=1, kickoff_timeout_s=0.0, goal_reward=0.0, win_reward=0.0,
                 idle_dist=40.0, idle_hold_s=5.0, idle_penalty=0.1, idle_table=got)
    env.engine.set_state(np.array([(100.0, 0.0)]), np.zeros((1, 2)),
                         np.array([[(-300.0, 0.0), (300.0, 0.0)]]), np.zeros((1, 2, 2)))
    env.state = env.engine.snapshot()
    env.idle_anchor[:] = (100.0, 0.0)
    env.score[0] = (0, 1)
    r = run(env, 6.0)
    assert np.allclose(-r[-1], (0.1, 0.0), atol=1e-6), -r[-1]


def test_off_by_default():
    env = HaxEnv(1, team_size=1)
    assert env.idle_pen == 0.0 and not env.frame(0)["idle_hot"]


if __name__ == "__main__":
    for t in (test_nearest_player_pays_after_5s, test_rolling_ball_is_free, test_moving_the_ball_resets_the_clock,
              test_touch_in_corner_box_resets, test_scale_zero_measures_but_charges_nothing, test_1v1_schedule,
              test_share_by_score_and_distance, test_preset_winner_never_pays, test_off_by_default):
        t()
        print("PASS", t.__name__)
