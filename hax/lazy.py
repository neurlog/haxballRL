"""Training-only "lazy" opponents, so the learner also sees opponents that don't contest the
ball, e.g. an open goal. (Self-play and league opponents always come for the ball, and the
1v1 champion was measured to almost never score when the opponent simply walks away.)

Each lazy game picks one behaviour for the whole opposing team:

    statue   stands still wherever it is (spawn after each kickoff)
    runaway  moves to the point of the pitch farthest from the ball
    wander   walks in a random direction, changing it every ~1 s

They never kick and never appear in the logs or win rates.
"""

from __future__ import annotations

import numpy as np

from .env import DIRS, HaxEnv

BEHAVIOURS = ("statue", "runaway", "wander")

_LUT = np.zeros((3, 3), np.int64)  # (dx+1, dy+1) -> direction index
for _i, (_x, _y) in enumerate(DIRS.tolist()):
    _LUT[_x + 1, _y + 1] = _i


class LazyOpponents:
    def __init__(self, env: HaxEnv, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.kind = np.zeros(env.n, np.int64)                       # behaviour index per env
        self.wander_dir = rng.integers(0, 9, (env.n, env.P))      # current random direction
        self.corners = None

    def reset_env(self, e: int):
        self.kind[e] = self.rng.integers(len(BEHAVIOURS))
        self.wander_dir[e] = self.rng.integers(0, 9, self.env.P)

    def actions(self, envs: np.ndarray, agents: np.ndarray) -> np.ndarray:
        """Joint actions (agent's mirrored frame) for the given (env, agent) pairs."""
        env, st = self.env, self.env.state
        if self.corners is None:  # far points of the pitch, from the stadium's goal line
            gx = env.goal_x * 0.92
            gy = gx * 0.42
            self.corners = np.array([[gx, gy], [gx, -gy], [-gx, gy], [-gx, -gy], [0, gy], [0, -gy]])
        out = np.zeros(len(envs), np.int64)
        kinds = self.kind[envs]
        for i, (e, a) in enumerate(zip(envs, agents)):
            k = BEHAVIOURS[kinds[i]]
            if k == "statue":
                continue  # direction 0 = stand still, no kick
            if k == "wander":
                if self.rng.random() < 1 / 30:  # ~once a second at 30 decisions/s
                    self.wander_dir[e, a] = self.rng.integers(0, 9)
                d = DIRS[self.wander_dir[e, a]].copy()
            else:  # runaway: head for the far point farthest from the ball
                tgt = self.corners[np.argmax(np.linalg.norm(self.corners - st.ball_pos[e], axis=1))]
                v = tgt - st.player_pos[e, a]
                d = np.array([int(np.sign(v[0])) if abs(v[0]) > 8 else 0, int(np.sign(v[1])) if abs(v[1]) > 8 else 0])
            d[0] *= int(env.sx[e, a])  # world -> the agent's mirrored frame
            out[i] = _LUT[d[0] + 1, d[1] + 1]
        return out
