"""The training-only lazy opponents behave as described (python tests/test_lazy.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hax.env import HaxEnv  # noqa: E402
from hax.lazy import BEHAVIOURS, LazyOpponents  # noqa: E402


def run(kind, seconds=6, stadium="classic", ts=1):
    env = HaxEnv(1, team_size=ts, stadium=stadium, ping_ms=0, kickoff_timeout_s=0)
    lazy = LazyOpponents(env, np.random.default_rng(0))
    lazy.kind[0] = BEHAVIOURS.index(kind)
    # put the ball in play near the middle-left so "away from the ball" is meaningful
    pp = env.state.player_pos.copy()
    env.engine.set_state(np.array([[-150.0, 20.0]]), np.zeros((1, 2)), pp, np.zeros_like(pp))
    env.state = env.engine.snapshot()
    blue = np.arange(ts, 2 * ts)
    start = env.state.player_pos[0, blue].copy()
    for _ in range(seconds * 30):
        a = np.zeros((1, env.P), np.int64)
        a[0, blue] = lazy.actions(np.zeros(ts, np.int64), blue)
        env.step(a)
    end = env.state.player_pos[0, blue]
    moved = float(np.linalg.norm(end - start, axis=1).mean())
    dball = float(np.linalg.norm(end - env.state.ball_pos[0], axis=1).mean())
    return moved, dball


def test_lazy_behaviours():
    m, _ = run("statue")
    print(f"  statue : moved {m:5.1f} px")
    assert m < 1.0
    m, d = run("runaway")
    print(f"  runaway: moved {m:5.1f} px, ends {d:5.0f} px from the ball")
    assert m > 50 and d > 400
    m, _ = run("wander")
    print(f"  wander : moved {m:5.1f} px")
    assert m > 20
    m, d = run("runaway", stadium="big", ts=3)
    print(f"  3v3 Big runaway team: moved {m:5.1f} px, ends {d:5.0f} px from the ball")
    assert m > 50 and d > 500


if __name__ == "__main__":
    test_lazy_behaviours()
    print("PASS test_lazy_behaviours")
