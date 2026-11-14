"""Hand-written opponents, vectorised over all envs. They read the true world state.

* `scripted_actions` - a decent rule-based player: holds its position when the opponent
  kicks off, drops back to defend when an opponent is closer to the ball in its own half,
  gets goal-side of the ball and pushes or kicks it towards the target goal, and circles
  around the ball when it is on the wrong side. 3% random actions.
* `chase_actions` - HaxballGym's baseline: run at the ball and kick when close.
"""

from __future__ import annotations

import numpy as np

from .env import DIRS, RED, HaxEnv


def _dir_to_action(v: np.ndarray, kick: np.ndarray, dead: float = 0.35) -> np.ndarray:
    """World-frame desired direction (..., 2) -> joint action in world frame."""
    nrm = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    u = v / nrm
    dx = np.where(u[..., 0] > dead, 1, np.where(u[..., 0] < -dead, -1, 0))
    dy = np.where(u[..., 1] > dead, 1, np.where(u[..., 1] < -dead, -1, 0))
    return _LUT[dx + 1, dy + 1] + 9 * kick.astype(np.int64)


_LUT = np.zeros((3, 3), np.int64)  # (dx+1, dy+1) -> direction index
for _i, (_x, _y) in enumerate(DIRS.tolist()):
    _LUT[_x + 1, _y + 1] = _i


def _to_agent_frame(a_world: np.ndarray, env: HaxEnv) -> np.ndarray:
    """Actions are fed to the env in each agent's mirrored frame: flip dx for blue."""
    d, k = a_world % 9, a_world // 9
    dxy = DIRS[d].copy()
    dxy[..., 0] *= env.sx.astype(np.int64)
    return _LUT[dxy[..., 0] + 1, dxy[..., 1] + 1] + 9 * k


def chase_actions(env: HaxEnv) -> np.ndarray:
    st = env.state
    d = st.ball_pos[:, None] - st.player_pos
    kick = np.linalg.norm(d, axis=-1) < 30
    return _to_agent_frame(_dir_to_action(d, kick), env)


def scripted_actions(env: HaxEnv, rng=None) -> np.ndarray:
    rng = env.rng if rng is None else rng
    st = env.state
    pp, ball = st.player_pos, st.ball_pos[:, None]  # (N,P,2), (N,1,2)
    sx = env.sx  # +1 red (attacks +x), -1 blue
    tgt = np.stack([sx * env.goal_x, np.zeros_like(sx)], -1)
    own = np.stack([-sx * env.goal_x, np.zeros_like(sx)], -1)
    to_ball = ball - pp
    dist = np.linalg.norm(to_ball, axis=-1)
    g = tgt - ball
    gdir = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
    behind = ball - gdir * 22.0  # spot just behind the ball, on the goal line of fire
    # am I on the right side of the ball? (between ball and own goal, roughly)
    along = ((pp - ball) * gdir).sum(-1)  # < 0 means behind the ball
    lateral = np.abs((pp - ball)[..., 0] * gdir[..., 1] - (pp - ball)[..., 1] * gdir[..., 0])
    aligned = (along < -5) & (lateral < 14)
    # circle round the ball when on the wrong side: aim for a point offset sideways
    side = np.sign((pp - ball)[..., 1] + 1e-6)[..., None]
    detour = ball - gdir * 30.0 + np.stack([np.zeros_like(side[..., 0]), side[..., 0]], -1) * 38.0
    target = np.where(aligned[..., None], ball + gdir * 10.0, np.where((along > -5)[..., None], detour, behind))
    # defend: an opponent is nearer the ball and the ball is in our half -> drop goal-side
    team = env.team
    opp_mask = team[:, None, :] != team[:, :, None]
    all_d = np.linalg.norm(ball - pp, axis=-1)  # (N,P)
    opp_best = np.where(opp_mask, all_d[:, None, :], np.inf).min(-1)
    in_our_half = (ball[..., 0] * sx) < 40
    defend = (opp_best + 25 < dist) & in_our_half
    guard = own + (ball - own) * 0.35
    target = np.where(defend[..., None], guard, target)
    # opponent kickoff: stay put (the barrier keeps us out anyway)
    ko = env.kickoff_waiting()[:, None] & (team != env.kicking[:, None])
    v = target - pp
    v = np.where(ko[..., None], 0.0, v)
    kick = (dist < 28) & (along < 0)
    a = _dir_to_action(v, kick)
    a = np.where(np.linalg.norm(v, axis=-1) < 3, 9 * kick, a)
    rnd = rng.random(a.shape) < 0.03
    a = np.where(rnd, rng.integers(0, 18, a.shape), a)
    return _to_agent_frame(a, env)


__all__ = ["scripted_actions", "chase_actions", "RED"]
