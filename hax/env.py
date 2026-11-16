"""Full HaxBall games on top of HaxballGym's bit-exact physics core.

HaxballGym (`haxball_core`) gives us the real engine: physics, kickoff barriers, the
150-tick goal celebration and the conceding-team re-kickoff. This module adds what a
*game* needs on top, all vectorised over N matches:

* **Simulated ping (optional, default 0).** Every input reaches the engine `ping_ms` after the agent saw the
  state (stochastic rounding of the fractional tick plus Gaussian jitter, never
  reordered). The engine runs at tick_skip=1 so the delay has single-tick resolution,
  and the agent decides every `frame_skip` ticks.
* **The match clock.** 3:00 of play; the clock is stopped while a kickoff is waiting
  for its first touch and during the goal celebration, like HaxBall. Level at 3:00 ->
  golden-goal overtime (capped, then a draw).
* **Training-only kickoff anti-stall.** If the kicking team hasn't touched a kickoff
  after `kickoff_timeout_s`, its nearest player is walked into the ball, so a passive
  (e.g. freshly initialised) opponent cannot freeze a game. 0 turns it off.
* **Observations** for an entity network (`SELF_DIM` + one `ENT_DIM` row per other
  player, teammates first), x-mirrored for blue so every agent attacks +x, positions
  scaled by the stadium's size (`stadium_scales`).
* **Rewards** (RLGym-style: every term is in [-1, 1] and its weight sets its influence):
  * events: +/-`goal_reward` per goal scored/conceded, +/-`win_reward` to every player of
    the winning/losing team when the game ends (0 for a draw). With `win_on_lead` the win
    reward is paid when the lead changes instead: +win_reward when your team goes ahead,
    -win_reward when it loses the lead (and the same from level). It adds up to exactly the
    end-of-game reward (+win/-win/0) but arrives when the deciding goal happens, so it isn't
    discounted away over a 3-minute game (gamma 0.995 is a ~7 s horizon);
  * dense, per decision, weights set by `set_reward_weights(speed, touch, ball_to_goal)`:
    - velocity towards the ball, signed (HaxballGym/RLGym VelocityPlayerToBall): moving away
      costs what moving closer earns, so stepping back and forth next to the ball earns ~0,
    - touch: the change in the ball's velocity TOWARDS THE GOAL YOU ATTACK while you are
      touching it, signed (a full kick straight at goal = +1, one straight away = -1).
      Credited to one player per team (the one nearest the ball), so a scrum earns one touch.
      Kicking the ball into a wall and having it bounce back sums to about zero, so
      pinning the ball in a corner and spamming kick earns nothing
      (tests/test_rewards.py checks this),
    - ball velocity towards the goal you attack (VelocityBallToGoalReward). With `btg_zero_sum`
      it's (yours - the opponents') / 2, so whatever one team gains the other loses.
    The velocity-to-ball term goes only to the player on each team nearest the ball
    (identical to before in 1v1), so in team games they don't all chase it in a swarm.
  * kickoff dawdling: once your own kickoff has waited `kickoff_grace_s` (5 s), the kicking
    team loses `kickoff_penalty` per decision until the ball moves (the clock is stopped
    during kickoffs, so without this waiting is free and agents learn to stall),
  * corner stalling (training-only, off by default): a `corner_box` x `corner_box` px square sits in
    each corner of the pitch. Once the ball has stayed inside one for `corner_hold_s` of play,
    EVERY player loses `corner_penalty` per decision until it leaves. Both sides pay the same, so
    pinning the ball in a corner is never safe, and it can only cost reward (nothing to farm).
    The squares are drawn see-through red in the viewers only; agents don't observe them,
  * idle ball (training-only, off by default): if the ball stays within `idle_dist` px of where it
    was for more than `idle_hold_s` of play, the player NEAREST the ball loses `idle_penalty` per
    decision until the ball moves `idle_dist` px. Run v6 learned to park both bots in their own
    goal mouths and leave the ball alone (98% 0-0). The nearest player is the one who can most
    easily move it, so this pushes play on without punishing the bot that is defending; tapping
    the ball in place doesn't reset it (it has to travel), dribbling and passing do. Inside a
    corner square any touch resets it too: corner play is left to the corner rule above.
    `idle_table` scales it by your team's score and whether you're the nearest player:
    rows winning / level / losing, columns (nearest, everyone else); the default (1, 0) on every
    row is "nearest player pays in full, nobody else",
  * team play (all off by default, used for 3v3), set by `set_team_rewards`:
    - pass: a teammate's touch followed by another teammate's touch, no opponent touch in
      between. Paid only if the ball travelled >= `pass_min_dist` (scaled up to 1 at 300 px)
      AND ended closer to the opponent goal than the passer's last touch (any direction with
      `pass_backward`). Zero-sum (the opponents lose it), with a 1 s cooldown per team. Corner
      ping-pong (short) earns nothing, and forward-only, neither does endless back-and-forth,
    - crowding: each pair of teammates closer than `crowd_dist` costs both
      `crowd_penalty` x (1 - d / crowd_dist) per decision,
    - team spirit (RLGym): individual dense terms (velocity to ball, touch, crowding) are
      blended with the team average: r = (1 - tau) r_self + tau mean_team(r).

Actions are one of 18 joint actions: 9 move directions x kick on/off.
"""

from __future__ import annotations

import numpy as np
from haxballgym import TransitionEngine

RED, BLUE = 2, 4
TICKS_PER_S = 60
PLAYER_TOP_SPEED = 2.4  # accel 0.1 * damping 0.96 / (1 - 0.96)
BALL_SPEED_REF = 6.0  # about the fastest a ball gets from a kick (kick strength 5 + player speed)

# move directions, HaxBall convention: +y is DOWN the screen
DIRS = np.array([(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)], np.int64)
N_ACTIONS = 18  # action = dir + 9 * kick

VEL_SCALE = 1 / 5.0


def stadium_scales(stadium: str):
    """Observation position scale (1/width, 1/height) and distance scale for a stadium, so
    positions land in about [-1, 1] on any map (like HaxballGym's DefaultObs 'auto'). On
    Classic (420 x 200) this is exactly 1/420, 1/200 and 400, as the 1v1 models were trained."""
    import json

    from haxballgym import stadium_text
    d = json.loads(stadium_text(stadium))
    w, h = float(d["width"]), float(d["height"])
    return np.array([1.0 / w, 1.0 / h]), 400.0 * w / 420.0
SELF_DIM = 27
ENT_DIM = 8


def field_size(stadium: str):
    """Half-width and half-height of the pitch lines (the 'bg' rectangle), e.g. 370 x 170 on Classic."""
    import json

    from haxballgym import stadium_text
    bg = json.loads(stadium_text(stadium)).get("bg", {})
    return float(bg.get("width", 370)), float(bg.get("height", 170))


def action_to_keys(a: np.ndarray) -> np.ndarray:
    """(...,) joint action -> (..., 3) engine keys (dx, dy, kick) in the agent's own frame."""
    a = np.asarray(a, np.int64)
    out = np.empty(a.shape + (3,), np.int64)
    out[..., :2] = DIRS[a % 9]
    out[..., 2] = a // 9
    return out


def keys_to_action(dx: int, dy: int, kick: bool) -> int:
    d = [tuple(r) for r in DIRS.tolist()].index((int(dx), int(dy)))
    return d + (9 if kick else 0)


class HaxEnv:
    def __init__(
        self,
        n_envs: int,
        team_size: int = 1,
        stadium: str = "classic",
        seed: int = 0,
        ping_ms: float = 0.0,
        jitter_ms: float = 0.0,
        frame_skip: int = 2,
        time_limit_s: float = 180.0,
        overtime_limit_s: float = 60.0,
        max_total_s: float = 330.0,
        kickoff_timeout_s: float = 10.0,
        goal_reward: float = 1.0,
        win_reward: float = 3.0,
        win_on_lead: bool = False,
        btg_zero_sum: bool = False,
        kickoff_penalty: float = 0.0,
        kickoff_grace_s: float = 5.0,
        corner_box: float = 0.0,
        corner_hold_s: float = 5.0,
        corner_penalty: float = 0.0,
        idle_dist: float = 50.0,
        idle_hold_s: float = 5.0,
        idle_penalty: float = 0.0,
        idle_table=((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)),
    ):
        self.n, self.ts = n_envs, team_size
        self.P = 2 * team_size
        self.engine = TransitionEngine(
            n_envs, team_size, team_size, step_limit=1 << 62, tick_skip=1,
            stadium=None if stadium == "classic" else stadium,
        )
        self.stadium = stadium
        self.pos_scale, self.dist_scale = stadium_scales(stadium)
        self.rng = np.random.default_rng(seed)
        self.ping_ticks = ping_ms * TICKS_PER_S / 1000.0
        self.jitter_ticks = jitter_ms * TICKS_PER_S / 1000.0
        self.frame_skip = frame_skip
        self.time_limit = int(time_limit_s * TICKS_PER_S)
        self.ot_limit = int(overtime_limit_s * TICKS_PER_S)
        self.max_total = int(max_total_s * TICKS_PER_S)
        self.ko_timeout = int(kickoff_timeout_s * TICKS_PER_S) if kickoff_timeout_s > 0 else 1 << 62
        self.goal_reward = goal_reward
        self.win_reward = win_reward
        self.win_on_lead = win_on_lead
        self.btg_zero_sum = btg_zero_sum
        self.ko_penalty = kickoff_penalty / frame_skip  # per tick, so `kickoff_penalty` per decision
        self.ko_grace = int(kickoff_grace_s * TICKS_PER_S)
        self.corner_box = corner_box
        self.corner_hold = int(corner_hold_s * TICKS_PER_S)
        self.corner_pen = corner_penalty / frame_skip  # per tick, so `corner_penalty` per decision
        self.corner_scale = 1.0  # set_stall_scales: 0 keeps measuring corner time but charges nothing
        self.field_w, self.field_h = field_size(stadium)
        self.idle_dist = idle_dist
        self.idle_hold = int(idle_hold_s * TICKS_PER_S)
        self.idle_pen = idle_penalty / frame_skip  # per tick, so `idle_penalty` per decision
        self.idle_scale = 1.0  # set_idle_scale: 0 keeps measuring idle time but charges nothing
        # share of idle_penalty by your team's score state and whether you're the nearest player:
        # rows winning / level / losing, columns (nearest, everyone else)
        self.idle_table = np.asarray(idle_table, np.float64).reshape(3, 2)
        self.w_speed = self.w_touch = self.w_btg = 0.0
        self.w_pass = self.w_crowd = self.team_spirit = 0.0
        self.pass_min_dist, self.crowd_dist = 100.0, 40.0
        self.pass_backward = False

        self.team = self.engine._teams.copy()  # (N, P)
        self.sx = np.where(self.team == RED, 1.0, -1.0)  # x-mirror per agent
        p0, p1, gt = self.engine._goal_p0, self.engine._goal_p1, self.engine._goal_team
        self.goal_x = float(np.abs(p0[:, 0]).max())
        self.goal_y = float(np.abs(np.concatenate([p0[:, 1], p1[:, 1]])).max())
        # in the mirrored frame every agent attacks +goal_x and defends -goal_x
        self.ent_order = self._entity_order()
        self.obs_dim = SELF_DIM + (self.P - 1) * ENT_DIM
        self.n_mates, self.n_opps = team_size - 1, team_size
        self.K = 6  # pending-input slots per agent
        self.reset()

    # ------------------------------------------------------------------ setup
    def _entity_order(self):
        """For each agent: indices of the other players, teammates first then opponents."""
        order = np.zeros((self.P, self.P - 1), np.int64)
        teams = self.team[0]
        for k in range(self.P):
            mates = [j for j in range(self.P) if j != k and teams[j] == teams[k]]
            opps = [j for j in range(self.P) if teams[j] != teams[k]]
            order[k] = mates + opps
        return order

    def reset(self) -> np.ndarray:
        self.engine.reset()
        n, P = self.n, self.P
        self.t = 0
        self.score = np.zeros((n, 2), np.int64)  # red, blue
        self.play_ticks = np.zeros(n, np.int64)
        self.total_ticks = np.zeros(n, np.int64)
        self.celeb = np.zeros(n, np.int64)
        self.ko_wait = np.zeros(n, np.int64)
        self.kicking = np.full(n, RED, np.int64)
        self.touches = np.zeros((n, P), np.int64)
        self.pend_t = np.full((n, P, self.K), -1, np.int64)
        self.pend_a = np.zeros((n, P, self.K, 3), np.int64)
        self.last_apply = np.zeros((n, P), np.int64)
        self.applied = np.zeros((n, P, 3), np.int64)
        self.last_act = np.zeros((n, P, 2), np.int64)
        self.last_toucher = np.full(n, -1, np.int64)  # for pass detection
        self.last_touch_pos = np.zeros((n, 2))
        self.pass_cool = np.zeros((n, 2), np.int64)  # per team (red, blue), ticks
        self.passes = np.zeros((n, 2), np.int64)  # rewarded passes this game, per team
        self.corner_ticks = np.zeros(n, np.int64)  # ticks of play the ball has been in a corner box
        self.corner_pen_ticks = np.zeros(n, np.int64)  # ticks of corner penalty this game
        self.idle_anchor = np.zeros((n, 2))  # where the ball was when it last travelled idle_dist
        self.idle_ticks = np.zeros(n, np.int64)  # ticks of play since then
        self.idle_pen_ticks = np.zeros(n, np.int64)  # ticks of idle penalty this game
        self.slot = 0
        self.state = self.engine.snapshot()
        return self.observe()

    def _reset_envs(self, mask: np.ndarray):
        self.engine.reset_mask(mask)
        self.score[mask] = 0
        self.play_ticks[mask] = 0
        self.total_ticks[mask] = 0
        self.celeb[mask] = 0
        self.ko_wait[mask] = 0
        self.kicking[mask] = RED
        self.touches[mask] = 0
        self.pend_t[mask] = -1
        self.applied[mask] = 0
        self.last_apply[mask] = self.t
        self.last_act[mask] = 0
        self.last_toucher[mask] = -1
        self.pass_cool[mask] = 0
        self.passes[mask] = 0
        self.corner_ticks[mask] = 0
        self.corner_pen_ticks[mask] = 0
        self.idle_anchor[mask] = 0.0
        self.idle_ticks[mask] = 0
        self.idle_pen_ticks[mask] = 0

    def set_idle_scale(self, scale: float):
        """Fraction (0..1) of `idle_penalty` actually charged; training switches it on after a while."""
        self.idle_scale = float(scale)

    def set_corner_scale(self, scale: float):
        """Fraction (0..1) of `corner_penalty` actually charged; training switches it on after a while."""
        self.corner_scale = float(scale)

    def set_reward_weights(self, speed: float = 0.0, touch: float = 0.0, ball_to_goal: float = 0.0):
        self.w_speed, self.w_touch, self.w_btg = speed, touch, ball_to_goal

    def set_team_rewards(self, pass_reward: float = 0.0, crowd_penalty: float = 0.0, team_spirit: float = 0.0,
                         pass_min_dist: float = 100.0, crowd_dist: float = 40.0, pass_backward: bool = False):
        self.w_pass, self.w_crowd, self.team_spirit = pass_reward, crowd_penalty, team_spirit
        self.pass_min_dist, self.crowd_dist = pass_min_dist, crowd_dist
        self.pass_backward = pass_backward

    # ------------------------------------------------------------------ step
    def step(self, actions: np.ndarray):
        """actions (N, P) joint actions in each agent's mirrored frame.
        Returns obs (N,P,D), rewards (N,P), done (N,), finished games
        [(env, red_goals, blue_goals, total_ticks, red_touches, blue_touches, rewarded_passes,
          corner_penalty_ticks, idle_penalty_ticks)]."""
        n, P = self.n, self.P
        actions = np.asarray(actions, np.int64).reshape(n, P)
        keys = action_to_keys(actions)
        keys[..., 0] *= self.sx.astype(np.int64)  # back to world frame
        # --- queue the inputs behind the simulated ping
        d = self.ping_ticks + self.rng.normal(0.0, self.jitter_ticks, (n, P))
        d = np.maximum(d, 0.0)
        d = np.floor(d) + (self.rng.random((n, P)) < (d - np.floor(d)))
        apply_t = np.maximum(self.t + d.astype(np.int64), self.last_apply)
        self.last_apply = apply_t
        s = self.slot % self.K
        self.slot += 1
        self.pend_t[:, :, s] = apply_t
        self.pend_a[:, :, s] = keys
        self.last_act = np.stack([actions, self.last_act[..., 0]], -1)

        rew = np.zeros((n, P), np.float32)  # team-level terms
        indiv = np.zeros((n, P), np.float64)  # individual terms, blended by team spirit
        touch = np.zeros((n, P), np.float64)
        done = np.zeros(n, bool)
        ended_goal = np.zeros(n, bool)
        for _ in range(self.frame_skip):
            due = (self.pend_t >= 0) & (self.pend_t <= self.t)
            if due.any():
                pick = np.argmax(np.where(due, self.pend_t, -1), axis=-1)  # newest due input
                anyd = due.any(-1)
                new = np.take_along_axis(self.pend_a, pick[..., None, None], axis=2)[:, :, 0]
                self.applied = np.where(anyd[..., None], new, self.applied)
                self.pend_t[due] = -1
            st0 = self.state
            waiting = (self.celeb == 0) & (np.abs(st0.ball_pos).sum(-1) == 0) & (np.abs(st0.ball_vel).sum(-1) == 0)
            applied = self.applied
            stalled = waiting & (self.ko_wait >= self.ko_timeout)
            if stalled.any():
                # training-only anti-stall: the kicking team's nearest player walks into the ball
                applied = applied.copy()
                for e in np.where(stalled)[0]:
                    cand = np.where(self.team[e] == self.kicking[e])[0]
                    dv = st0.ball_pos[e] - st0.player_pos[e, cand]
                    k = cand[np.argmin(np.linalg.norm(dv, axis=-1))]
                    d = st0.ball_pos[e] - st0.player_pos[e, k]
                    applied[e, k] = (int(np.sign(np.round(d[0]))), int(np.sign(np.round(d[1]))), 1)
            scored = self.engine._e.physics_step(np.ascontiguousarray(applied))
            self.t += 1
            self.total_ticks += 1
            self.state = st = self.engine.snapshot()
            # clock: stopped during the celebration and while a kickoff waits for its touch
            self.ko_wait = np.where(waiting, self.ko_wait + 1, 0)
            if self.ko_penalty:
                # dawdling penalty: once your own kickoff has waited `kickoff_grace_s`, every
                # player of the kicking team loses `kickoff_penalty` per decision until the ball moves
                late = waiting & (self.ko_wait > self.ko_grace) & ~done
                if late.any():
                    rew -= np.where(late[:, None] & (self.team == self.kicking[:, None]), self.ko_penalty, 0.0
                                    ).astype(np.float32)
            running = (self.celeb == 0) & ~waiting
            self.play_ticks += running & ~done
            if self.corner_box > 0:
                b = np.abs(st.ball_pos)
                inside = (b[:, 0] > self.field_w - self.corner_box) & (b[:, 1] > self.field_h - self.corner_box)
                self.corner_ticks = np.where(inside & (self.celeb == 0), self.corner_ticks + running, 0)
                hot = (self.corner_ticks > self.corner_hold) & running & ~done
                self.corner_pen_ticks += hot
                if self.corner_pen and self.corner_scale > 0 and hot.any():
                    rew -= np.where(hot[:, None], self.corner_pen * self.corner_scale, 0.0).astype(np.float32)
            if self.idle_pen:
                moved = (np.linalg.norm(st.ball_pos - self.idle_anchor, axis=-1) > self.idle_dist) | (self.celeb > 0)
                if self.corner_box > 0:  # a touch inside a corner square counts as moving the ball
                    b = np.abs(st.ball_pos)
                    in_box = (b[:, 0] > self.field_w - self.corner_box) & (b[:, 1] > self.field_h - self.corner_box)
                    touched = np.linalg.norm(st.ball_pos[:, None] - st.player_pos, axis=-1).min(-1) < 26.5
                    moved |= in_box & touched
                self.idle_anchor[moved] = st.ball_pos[moved]
                self.idle_ticks = np.where(moved, 0, self.idle_ticks + running)
                idle = (self.idle_ticks > self.idle_hold) & running & ~done
                self.idle_pen_ticks += idle
                if idle.any() and self.idle_scale > 0:
                    near = np.linalg.norm(st.ball_pos[:, None] - st.player_pos, axis=-1).argmin(-1)
                    is_near = np.zeros((n, P), np.int64)
                    is_near[np.arange(n), near] = 1
                    lead = (self.score[:, 0] - self.score[:, 1])[:, None] * np.where(self.team == RED, 1, -1)
                    row = np.where(lead > 0, 0, np.where(lead == 0, 1, 2))  # winning / level / losing
                    share = self.idle_table[row, 1 - is_near]
                    rew -= np.where(idle[:, None], self.idle_pen * self.idle_scale * share, 0.0).astype(np.float32)
            self.celeb = np.maximum(self.celeb - 1, 0)
            g = scored >= 0
            if g.any():
                conc = scored.astype(np.int64)
                red_scored = g & (conc == BLUE)
                blue_scored = g & (conc == RED)
                gi = np.where(g & ~done)[0]
                lead0 = np.sign(self.score[gi, 0] - self.score[gi, 1])
                self.score[:, 0] += red_scored & ~done
                self.score[:, 1] += blue_scored & ~done
                sign = np.where(self.team[gi] == np.where(red_scored[gi], RED, BLUE)[:, None], 1.0, -1.0)
                rew[gi] += self.goal_reward * sign
                if self.win_on_lead:  # red's lead went from lead0 to lead1; blue gets the opposite
                    dlead = np.sign(self.score[gi, 0] - self.score[gi, 1]) - lead0
                    side = np.where(self.team[gi] == RED, 1.0, -1.0)
                    rew[gi] += (self.win_reward * dlead[:, None] * side).astype(np.float32)
                self.celeb[g] = 150
                self.kicking[g] = conc[g]
                self.last_toucher[g] = -1
                # golden goal: a goal in overtime ends the game straight away
                ended_goal |= g & (self.play_ticks >= self.time_limit)
            dist0 = np.linalg.norm(st0.ball_pos[:, None] - st0.player_pos, axis=-1)
            dist = np.linalg.norm(st.ball_pos[:, None] - st.player_pos, axis=-1)
            self.touches += dist < 26.5
            if self.w_touch or self.w_pass:
                # velocity change the ball got this tick beyond its normal damping (0.99/tick)
                dv = st.ball_vel - 0.99 * st0.ball_vel  # (N,2)
                contact = (np.minimum(dist0, dist) < 30.0) & ((self.celeb == 0) & ~stalled & ~g)[:, None]
            if self.w_touch:
                # projected on the direction to the goal each player attacks, credited WITH SIGN
                # to the player in contact (before or after the tick). Kick + wall rebound ~ 0.
                # Only ONE player per team is credited - the one nearest the ball - so a scrum of
                # teammates around the ball earns one touch, not three (identical in 1v1).
                gain = (self._goal_dir(st0.ball_pos) * dv[:, None]).sum(-1) / 5.0  # (N,P)
                dc = np.where(contact, dist0, np.inf).reshape(-1, 2, self.ts)
                credited = (contact.reshape(-1, 2, self.ts) & (dc == dc.min(-1, keepdims=True))).reshape(contact.shape)
                touch += np.where(credited, gain, 0.0)
            if self.w_pass:
                rew += self._passes(st, dist0, contact, np.linalg.norm(dv, axis=-1), done)
            done |= ended_goal
            over = self.play_ticks >= self.time_limit
            level = self.score[:, 0] == self.score[:, 1]
            done |= over & ~level & (self.celeb == 0)
            done |= self.play_ticks >= self.time_limit + self.ot_limit
            done |= self.total_ticks >= self.max_total

        if self.w_speed or self.w_btg:
            speed_r, btg_r = self._dense(self.state)
            indiv += speed_r
            rew += btg_r
        if self.w_touch:
            indiv += self.w_touch * np.clip(touch, -1.0, 1.0)
        if self.w_crowd and self.ts > 1:
            indiv -= self._crowding(self.state)
        if self.team_spirit and self.ts > 1:
            team_mean = indiv.reshape(n, 2, self.ts).mean(-1, keepdims=True)  # red players first, then blue
            team_mean = np.broadcast_to(team_mean, (n, 2, self.ts)).reshape(n, P)
            indiv = (1.0 - self.team_spirit) * indiv + self.team_spirit * team_mean
        rew += indiv.astype(np.float32)

        finished = []
        if done.any():
            idx = np.where(done)[0]
            # end-of-game reward: win > loss, on top of the per-goal rewards
            margin = np.sign(self.score[idx, 0] - self.score[idx, 1])  # +1 red won, -1 blue won, 0 draw
            side = np.where(self.team[idx] == RED, 1.0, -1.0)
            if not self.win_on_lead:  # (with win_on_lead it was already paid as the lead changed)
                rew[idx] += (self.win_reward * margin[:, None] * side).astype(np.float32)
            tch = self.touches[idx]
            for j, e in enumerate(idx):
                finished.append((int(e), int(self.score[e, 0]), int(self.score[e, 1]), int(self.total_ticks[e]),
                                 int(tch[j][self.team[e] == RED].sum()), int(tch[j][self.team[e] == BLUE].sum()),
                                 int(self.passes[e].sum()), int(self.corner_pen_ticks[e]),
                                 int(self.idle_pen_ticks[e])))
            self._reset_envs(done)
            self.state = self.engine.snapshot()
        return self.observe(), rew, done, finished

    def _dense(self, st):
        """Per-decision dense rewards (only while the ball is live), each in [-1, 1]:
        * velocity towards the ball, signed, for the player on each team nearest the ball
          (non-zero-sum; keeps both teams engaged, which is what stops the 'nobody touches
          the ball' collapse) -> individual,
        * ball velocity towards the goal you attack (zero-sum-ish) -> team-level."""
        live = (self.celeb == 0)[:, None]
        pb = st.ball_pos[:, None] - st.player_pos
        d = np.linalg.norm(pb, axis=-1)
        pbn = pb / np.maximum(d, 1e-9)[..., None]
        speed = np.clip((pbn * st.player_vel).sum(-1) / PLAYER_TOP_SPEED, -1.0, 1.0)
        dt = d.reshape(-1, 2, self.ts)  # red players first, then blue
        nearest = (dt == dt.min(-1, keepdims=True)).reshape(d.shape)
        btg = np.clip((self._goal_dir(st.ball_pos) * st.ball_vel[:, None]).sum(-1) / BALL_SPEED_REF, -1.0, 1.0)
        if self.btg_zero_sum:
            # (yours - the other team's) / 2: in a corner both goal mouths lie partly towards the
            # middle, so the ball leaving a corner would otherwise pay BOTH teams (farmed in run v4)
            bt = btg.reshape(-1, 2, self.ts)  # every player of a team attacks the same goal
            btg = ((bt - bt[:, ::-1]) / 2).reshape(btg.shape)
        return (np.where(live & nearest, self.w_speed * speed, 0.0),
                np.where(live, self.w_btg * btg, 0.0).astype(np.float32))

    def _crowding(self, st) -> np.ndarray:
        """(N,P) penalty: sum over teammates closer than crowd_dist of w_crowd * (1 - d / crowd_dist)."""
        pp = st.player_pos.reshape(-1, 2, self.ts, 2)
        d = np.linalg.norm(pp[:, :, :, None] - pp[:, :, None], axis=-1)  # (N,2,ts,ts)
        close = np.clip(1.0 - d / self.crowd_dist, 0.0, 1.0)
        close[..., np.arange(self.ts), np.arange(self.ts)] = 0.0
        live = (self.celeb == 0)[:, None, None]
        return np.where(live, self.w_crowd * close.sum(-1), 0.0).reshape(-1, self.P)

    def _passes(self, st, dist0, contact, dv_norm, done) -> np.ndarray:
        """Detect completed passes on this tick and return the (N,P) zero-sum pass reward."""
        out = np.zeros((self.n, self.P), np.float32)
        self.pass_cool = np.maximum(self.pass_cool - 1, 0)
        hit = contact & (dv_norm > 0.3)[:, None]  # a touch that actually changed the ball's motion
        envs = np.where(hit.any(-1) & ~done)[0]
        for e in envs:
            j = int(np.argmin(np.where(hit[e], dist0[e], np.inf)))  # nearest player in contact
            i = int(self.last_toucher[e])
            team_j = 0 if self.team[e, j] == RED else 1
            if i >= 0 and i != j and self.team[e, i] == self.team[e, j] and self.pass_cool[e, team_j] == 0:
                delta = st.ball_pos[e] - self.last_touch_pos[e]
                length = float(np.hypot(*delta))
                forward = delta[0] * (1.0 if team_j == 0 else -1.0)
                if length >= self.pass_min_dist and (forward >= 0.0 or self.pass_backward):
                    r = self.w_pass * min(length / 300.0, 1.0)
                    out[e] = np.where(self.team[e] == self.team[e, j], r, -r)
                    self.pass_cool[e, team_j] = TICKS_PER_S
                    self.passes[e, team_j] += 1
            self.last_toucher[e] = j
            self.last_touch_pos[e] = st.ball_pos[e]
        return out

    def _goal_dir(self, ball_pos: np.ndarray) -> np.ndarray:
        """(N,2) ball positions -> (N,P,2) unit vectors from the ball to the nearest point of
        the goal mouth each player attacks."""
        tgt = np.stack([self.sx * self.goal_x, np.clip(ball_pos[:, None, 1], -self.goal_y, self.goal_y)
                        * np.ones_like(self.sx)], -1)
        bg = tgt - ball_pos[:, None]
        return bg / np.maximum(np.linalg.norm(bg, axis=-1, keepdims=True), 1e-9)

    # ------------------------------------------------------------------ obs
    def observe(self) -> np.ndarray:
        st = self.state
        n, P = self.n, self.P
        sx = self.sx[..., None]
        flip = np.array([1.0, 1.0])
        pp = st.player_pos * np.concatenate([sx, np.ones_like(sx)], -1)  # mirrored positions
        pv = st.player_vel * np.concatenate([sx, np.ones_like(sx)], -1)
        mir = np.concatenate([sx, np.ones_like(sx)], -1)
        bp = st.ball_pos[:, None] * mir  # (N,P,2) ball in each agent's frame
        bv = st.ball_vel[:, None] * mir
        tgt = np.array([self.goal_x, 0.0]) * flip
        own = np.array([-self.goal_x, 0.0])
        out = np.zeros((n, P, self.obs_dim), np.float32)
        rel_b = bp - pp
        o = out
        o[..., 0:2] = pp * self.pos_scale
        o[..., 2:4] = pv * VEL_SCALE
        o[..., 4:6] = rel_b * self.pos_scale
        o[..., 6:8] = bv * VEL_SCALE
        o[..., 8:10] = bp * self.pos_scale
        o[..., 10:12] = (tgt - pp) * self.pos_scale
        o[..., 12:14] = (own - pp) * self.pos_scale
        o[..., 14] = np.linalg.norm(rel_b, axis=-1) / self.dist_scale
        k = action_to_keys(self.last_act)  # (N,P,2,3), own frame
        o[..., 15:21] = k.reshape(n, P, 6)
        waiting = ((self.celeb == 0) & (np.abs(st.ball_pos).sum(-1) == 0) & (np.abs(st.ball_vel).sum(-1) == 0))
        o[..., 21] = waiting[:, None]
        o[..., 22] = waiting[:, None] & (self.team == self.kicking[:, None])
        o[..., 23] = (self.celeb > 0)[:, None]
        o[..., 24] = np.minimum(self.play_ticks / self.time_limit, 1.0)[:, None]
        o[..., 25] = (self.play_ticks >= self.time_limit)[:, None]
        diff = (self.score[:, 0] - self.score[:, 1])[:, None] * np.where(self.team == RED, 1, -1)
        o[..., 26] = np.tanh(diff / 2.0)
        # other players (teammates first, then opponents)
        for kk in range(P):
            for slot, j in enumerate(self.ent_order[kk]):
                b = SELF_DIM + slot * ENT_DIM
                pj = st.player_pos[:, j] * mir[:, kk]
                vj = st.player_vel[:, j] * mir[:, kk]
                rel = pj - pp[:, kk]
                o[:, kk, b:b + 2] = rel * self.pos_scale
                o[:, kk, b + 2:b + 4] = vj * VEL_SCALE
                o[:, kk, b + 4:b + 6] = (bp[:, kk] - pj) * self.pos_scale
                o[:, kk, b + 6] = np.linalg.norm(rel, axis=-1) / self.dist_scale
                o[:, kk, b + 7] = np.linalg.norm(bp[:, kk] - pj, axis=-1) / self.dist_scale
        return out

    # ------------------------------------------------------------------ helpers
    def kickoff_waiting(self) -> np.ndarray:
        st = self.state
        return (self.celeb == 0) & (np.abs(st.ball_pos).sum(-1) == 0) & (np.abs(st.ball_vel).sum(-1) == 0)

    def frame(self, e: int) -> dict:
        st = self.state
        return dict(
            ball=st.ball_pos[e].tolist(), players=st.player_pos[e].tolist(), teams=self.team[e].tolist(),
            kicking=(self.applied[e, :, 2] > 0).tolist(), score=self.score[e].tolist(),
            time_s=self.play_ticks[e] / TICKS_PER_S, overtime=bool(self.play_ticks[e] >= self.time_limit),
            corner_box=self.corner_box, corner_hot=bool(self.corner_ticks[e] > self.corner_hold),
            idle_hot=bool(self.idle_pen and self.idle_ticks[e] > self.idle_hold),
        )
