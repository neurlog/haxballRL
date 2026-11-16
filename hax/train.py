"""PPO with an AlphaStar-style league on HaxballGym physics.

Each env slot has a fixed role, so the number of learner agents never changes:

    self    both teams are the learner (two learner agents per game)
    champ   learner vs the frozen champion (supplies the promotion statistics)
    league  learner vs a PFSP-sampled league member (resampled every game)

Default mix: 40% self-play, 30% champion, 25% league, 5% lazy opponents (training-only:
statue / runaway / wander, see hax/lazy.py; never logged).

The learner swaps colour every game. Every `promote_every` games the learner is promoted
to champion if it beat the champion >= 70% over >= 60 games and averages >= 50% against
the league's past champions and snapshots. The games vs the champion are either the 'champ'
training slots, or (with --promote-eval-games N) N separate evaluation games played at each
check without learning, as the Orbit Wars winner did; then --frac-champ can be 0.

Used by the per-mode folders, e.g. `1v1-classic/train.py`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .env import ENT_DIM, N_ACTIONS, SELF_DIM, HaxEnv
from .lazy import LazyOpponents
from .league import League
from .model import HaxNet, load_policy

MODES = ("self", "champ", "league")


def reward_weights(args, games):
    """Dense reward weights at this point in training: (speed, touch, ball_to_goal, team_spirit, stage, fade).
    * 0 -> reward_stage_games: everything moves linearly from its 'early' to its 'late' value;
    * push_fade_start -> + push_fade_games: touch and ball-to-goal fade linearly to zero (they reward pushing
      the ball forward, not controlling it). With --fade-chase the velocity-to-ball reward fades with them, so
      after the fade only goals, wins and the kickoff penalty are left (otherwise it stays, but it can then be
      farmed: two agents glued to the ball collect ~+11 per game each, more than a win at +3)."""
    stage = min(1.0, games / args.reward_stage_games) if args.reward_stage_games > 0 else 1.0
    lerp = lambda a, b: a + (b - a) * stage  # noqa: E731
    fade = 1.0
    if args.push_fade_games > 0:
        fade = min(1.0, max(0.0, 1.0 - (games - args.push_fade_start) / args.push_fade_games))
    return (lerp(args.speed_early, args.speed_late) * (fade if args.fade_chase else 1.0),
            lerp(args.touch_early, args.touch_late) * fade,
            lerp(args.btg_early, args.btg_late) * (1.0 if args.keep_btg else fade), lerp(args.team_spirit_early, args.team_spirit_late), stage, fade)


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--name", default="main", help="run name; saved to <mode folder>/runs/<name>")
    a("--runs-dir", default="runs")
    a("--resume", action="store_true")
    a("--init-from", default=None, help="start from these weights (e.g. a 1v1 checkpoint for 3v3)")
    a("--resume-weights", default=None, help="with --resume: keep the run's game count and league, but "
      "replace the learner with these weights (e.g. an earlier champion) and reset the optimiser")
    a("--total-games", type=int, default=10**9)
    # game
    a("--team-size", type=int, default=1)
    a("--stadium", default="classic")
    a("--ping-ms", type=float, default=0.0, help="simulated input latency for every agent")
    a("--jitter-ms", type=float, default=0.0)
    a("--frame-skip", type=int, default=2, help="physics ticks per decision (2 = 30 decisions/s)")
    a("--time-limit", type=float, default=180.0)
    a("--overtime-limit", type=float, default=60.0)
    a("--kickoff-timeout", type=float, default=10.0, help="training-only: walk the kicking player into the ball if a kickoff stalls this long (s); 0 = off")
    # reward
    a("--goal-reward", type=float, default=1.0, help="+ for each goal scored, - for each conceded")
    a("--win-reward", type=float, default=3.0, help="+ for winning the game, - for losing it, 0 for a draw")
    a("--win-on-lead", action="store_true", help="pay the win reward when the lead changes (sums to the same "
      "+win/-win/0 per game, but isn't discounted away by the end of a long game)")
    # RLGym-style dense rewards, per decision (30/s), each term in [0,1] or [-1,1]. Weights move
    # linearly from the 'early' to the 'late' value over --reward-stage-games, like the RLGym
    # guide's stages: first learn to reach and touch the ball, then shift to moving it to goal.
    # Nothing is removed completely. Set all to 0 for goals + wins only.
    a("--speed-early", type=float, default=0.02, help="velocity towards the ball (signed)")
    a("--speed-late", type=float, default=0.002)
    a("--touch-early", type=float, default=0.2, help="touch: change in ball velocity towards the goal (signed)")
    a("--touch-late", type=float, default=0.02)
    a("--btg-early", type=float, default=0.01, help="ball velocity towards the goal you attack")
    a("--btg-late", type=float, default=0.01)
    a("--reward-stage-games", type=float, default=3000)
    a("--push-fade-start", type=float, default=10000, help="games after which the touch and ball-to-goal rewards "
      "start fading to zero, so goals and wins decide everything")
    a("--push-fade-games", type=float, default=10000, help="games the fade takes (0 = no fade)")
    a("--btg-zero-sum", action="store_true", help="ball-to-goal as (yours - the opponents') / 2, so it can't pay both")
    a("--keep-btg", action="store_true", help="don't fade the ball-to-goal reward (it's zero-sum-ish, so it can't be farmed by both)")
    a("--fade-chase", action="store_true", help="fade the velocity-to-ball reward to zero with touch and ball-to-goal")
    a("--kickoff-penalty", type=float, default=0.002, help="per decision while your own kickoff waits past the grace time")
    a("--kickoff-grace", type=float, default=5.0, help="seconds of free waiting at your own kickoff")
    a("--corner-box", type=float, default=0.0, help="training-only: side (px) of the stall square in each corner; 0 = off")
    a("--corner-hold", type=float, default=5.0, help="seconds of play the ball may sit in a corner square for free")
    a("--corner-penalty", type=float, default=0.0, help="per decision, to EVERY player, while the ball overstays a corner square")
    a("--idle-penalty", type=float, default=0.0, help="training-only: per decision, to the player NEAREST the ball, "
      "while the ball has stayed within --idle-dist px for more than --idle-hold s of play; 0 = off")
    a("--idle-dist", type=float, default=50.0, help="the ball has to travel this far (px) to count as moving")
    a("--idle-hold", type=float, default=5.0, help="seconds of play the ball may sit still for free")
    a("--idle-table", default="1,0,1,0,1,0", help="share of --idle-penalty by score and distance: "
      "winning nearest,others, level nearest,others, losing nearest,others (6 numbers)")
    a("--idle-start-games", type=float, default=0, help="the idle penalty is off until this many games "
      "(idle time is still logged), then ramps up to full over --idle-ramp-games")
    a("--idle-ramp-games", type=float, default=0)
    a("--corner-start-games", type=float, default=0, help="the corner penalty is off until this many games "
      "(corner time is still logged), then ramps up to full over --corner-ramp-games")
    a("--corner-ramp-games", type=float, default=0)
    # team play (team games only; no effect in 1v1). See hax/env.py for the exploit guards.
    a("--pass-reward", type=float, default=0.0, help="zero-sum reward for a forward pass >= --pass-min-dist")
    a("--pass-min-dist", type=float, default=100.0)
    a("--pass-backward", action="store_true", help="backward passes count too (default: forward or sideways only)")
    a("--crowd-penalty", type=float, default=0.0, help="per decision, per teammate closer than --crowd-dist")
    a("--crowd-dist", type=float, default=40.0, help="teammates closer than this (px) count as crowding")
    a("--team-spirit-early", type=float, default=0.0, help="blend of own vs team-average individual rewards")
    a("--team-spirit-late", type=float, default=0.0)
    # ppo
    a("--n-envs", type=int, default=192)
    a("--rollout", type=int, default=192, help="decisions per env per update (~50k samples, as RLGym-PPO)")
    a("--gamma", type=float, default=0.995, help="~7 s horizon at 30 decisions/s")
    a("--lam", type=float, default=0.95)
    a("--lr", type=float, default=3e-4)
    a("--epochs", type=int, default=3)
    a("--minibatches", type=int, default=8)
    a("--clip", type=float, default=0.2)
    a("--vf-coef", type=float, default=0.5)
    a("--ent-coef", type=float, default=0.01)
    a("--ent-coef-final", type=float, default=0.002, help="entropy bonus after --ent-anneal-games")
    a("--ent-anneal-games", type=float, default=5000)
    a("--max-grad", type=float, default=0.5)
    a("--target-kl", type=float, default=0.03)
    a("--hidden", type=int, default=256)
    a("--ent-hidden", type=int, default=128)
    a("--threads", type=int, default=4)
    a("--device", default="auto", help="learner device: auto (cuda if available), cpu, cuda, cuda:0 ...")
    a("--opp-device", default="auto", help="frozen opponents' device: auto = the 2nd GPU if there are two, else --device")
    a("--seed", type=int, default=0)
    # league
    a("--frac-champ", type=float, default=0.30)
    a("--frac-league", type=float, default=0.25)
    a("--frac-lazy", type=float, default=0.05, help="training-only lazy opponents (statue / runaway / wander), "
      "taken from the self-play share; never logged")
    a("--promote-every", type=int, default=500)
    a("--promote-eval-games", type=int, default=0, help="at each promotion check, play this many separate "
      "evaluation games (no learning) vs the champion and promote on those; 0 = use the training games "
      "vs the champion (--frac-champ) instead")
    a("--promote-threshold", type=float, default=0.70)
    a("--promote-min-games", type=int, default=60)
    a("--promote-min-league", type=float, default=0.5)
    a("--snapshot-every-games", type=int, default=1000)
    a("--pfsp-topk", type=int, default=10)
    a("--pfsp-explore", type=float, default=0.1)
    a("--pfsp-power", type=float, default=2.0)
    a("--league-size", type=int, default=60)
    # viewer / logging
    a("--view-every", type=int, default=1000)
    a("--view-seconds", type=float, default=20.0)
    a("--view-skip", type=float, default=90.0, help="fast-forward the viewer game to this match time (s) first")
    a("--no-view", action="store_true")
    a("--log-every", type=int, default=10)
    a("--ckpt-every-games", type=int, default=1000)
    a("--max-minutes", type=float, default=0, help="stop cleanly (and save) after this long; 0 = no limit. "
      "Use it on Kaggle so a session ends before the 12 h limit")
    if defaults:
        p.set_defaults(**defaults)
    return p


def eval_vs(learner, opponent, env_kwargs, n_games, seed=0, dev="cpu", opp_dev="cpu"):
    """Play `n_games` evaluation games (no learning) of `learner` vs `opponent`, the learner red
    in half of them. Returns one result per game from the learner's side: 1 win, 0.5 draw, 0 loss."""
    env = HaxEnv(n_games, seed=seed, **env_kwargs)
    P, D, ts = env.P, env.obs_dim, env.ts
    nm, no = env.n_mates, env.n_opps
    red = np.arange(n_games) % 2 == 0
    li = np.array([e * P + (0 if red[e] else ts) + k for e in range(n_games) for k in range(ts)])
    oi = np.array([e * P + (ts if red[e] else 0) + k for e in range(n_games) for k in range(ts)])
    obs = env.observe().reshape(n_games * P, D)
    done, res = np.zeros(n_games, bool), [None] * n_games
    while not done.all():
        o = torch.from_numpy(obs)
        acts = np.zeros(n_games * P, np.int64)
        acts[li] = learner.act(o[li].to(dev), nm, no)[0].cpu().numpy()
        acts[oi] = opponent.act(o[oi].to(opp_dev), nm, no)[0].cpu().numpy()
        nobs, _, _, fin = env.step(acts.reshape(n_games, P))
        obs = nobs.reshape(n_games * P, D)
        for e, rg, bg, *_ in fin:
            if not done[e]:  # (finished games restart; only the first one counts)
                done[e] = True
                mine, theirs = (rg, bg) if red[e] else (bg, rg)
                res[e] = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return res


def idle_table(spec: str):
    """'a,b,c,d,e,f' -> ((a, b), (c, d), (e, f)): winning / level / losing x (nearest, others)."""
    v = [float(x) for x in str(spec).split(",")]
    if len(v) != 6:
        raise SystemExit(f"--idle-table needs 6 numbers, got {spec!r}")
    return tuple(zip(v[0::2], v[1::2]))


def ramp(games, start, length):
    """0 before `start` games, then linearly up to 1 over `length` games."""
    return min(1.0, max(0.0, (games - start) / max(length, 1.0)))


def refresh_opponents(env_opp, league) -> bool:
    """Re-pick the league opponent of every game whose member was just evicted (the league is
    capped at --league-size; run v6 crashed at 49k games loading an evicted snapshot).
    Returns True if anything changed."""
    live = {m["id"] for m in league.members}
    changed = False
    for e, mid in enumerate(env_opp):
        if mid is not None and mid not in live:
            env_opp[e] = league.sample()
            changed = True
    return changed


def fmean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def main(argv=None, defaults=None, mode_dir="."):
    args = build_parser(defaults).parse_args(argv)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = os.path.join(mode_dir, args.runs_dir, args.name)
    if os.path.exists(os.path.join(run_dir, "latest.pt")) and not args.resume:
        sys.exit(f"{run_dir} already has a run. Use --resume to continue it, or pick another --name.")
    os.makedirs(run_dir, exist_ok=True)

    N, ts = args.n_envs, args.team_size
    # the game itself (rules, clock, ping); the evaluation games use the same settings
    env_kwargs = dict(team_size=ts, stadium=args.stadium, ping_ms=args.ping_ms, jitter_ms=args.jitter_ms,
                      frame_skip=args.frame_skip, time_limit_s=args.time_limit,
                      overtime_limit_s=args.overtime_limit, kickoff_timeout_s=args.kickoff_timeout)
    env = HaxEnv(N, seed=args.seed, **env_kwargs,
                 goal_reward=args.goal_reward, win_reward=args.win_reward, win_on_lead=args.win_on_lead,
                 btg_zero_sum=args.btg_zero_sum,
                 kickoff_penalty=args.kickoff_penalty, kickoff_grace_s=args.kickoff_grace,
                 corner_box=args.corner_box, corner_hold_s=args.corner_hold, corner_penalty=args.corner_penalty,
                 idle_dist=args.idle_dist, idle_hold_s=args.idle_hold, idle_penalty=args.idle_penalty,
                 idle_table=idle_table(args.idle_table))
    P, D = env.P, env.obs_dim
    nm, no = env.n_mates, env.n_opps
    mkw = dict(self_dim=SELF_DIM, ent_dim=ENT_DIM, n_actions=N_ACTIONS, hidden=args.hidden, ent_hidden=args.ent_hidden)
    cuda = torch.cuda.is_available()
    dev = torch.device(("cuda:0" if cuda else "cpu") if args.device == "auto" else args.device)
    if args.opp_device == "auto":
        opp_dev = torch.device("cuda:1") if cuda and torch.cuda.device_count() > 1 else dev
    else:
        opp_dev = torch.device(args.opp_device)
    model = HaxNet(**mkw).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    ctor = lambda: HaxNet(**mkw)  # noqa: E731
    league = League(run_dir, ctor, topk=args.pfsp_topk, explore=args.pfsp_explore, power=args.pfsp_power,
                    max_size=args.league_size, device=opp_dev)
    snap = lambda: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # noqa: E731
    if cuda:
        print(f"devices: learner {dev}, opponents {opp_dev} ({torch.cuda.device_count()} GPUs: "
              + ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) + ")")

    games, updates, rows = 0, 0, []
    champ_results = collections.deque(maxlen=200)
    if args.resume:
        ck = torch.load(os.path.join(run_dir, "latest.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        games, updates = ck["games"], ck["updates"]
        champ_results.extend(ck.get("champ_results", []))
        league.load()
        if os.path.exists(os.path.join(run_dir, "log.csv")):
            with open(os.path.join(run_dir, "log.csv")) as f:
                rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        print(f"resumed {run_dir} at {games} games, champion gen {league.champion_gen}")
        if args.resume_weights:
            model.load_state_dict(load_policy(args.resume_weights).state_dict())
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
            champ_results.clear()
            print(f"learner weights replaced by {args.resume_weights} (optimiser reset)")
    else:
        if args.init_from:
            src = load_policy(args.init_from)
            model.load_state_dict(src.state_dict())
            print(f"initialised from {args.init_from}")
        else:
            # from scratch: let the network see its teammates from the start (the zero-init of
            # the teammate encoder only exists so a transferred 1v1 policy starts unchanged)
            model.mate_enc[2].reset_parameters()
        league.set_champion(snap(), 0, 0)
    with open(os.path.join(run_dir, "config.json"), "w") as f:  # the settings now in force (resumes too)
        json.dump(vars(args), f, indent=1)

    # ---- env slot roles (fixed), learner colour flips every game
    n_champ = int(round(N * args.frac_champ))
    n_league = int(round(N * args.frac_league))
    n_lazy = int(round(N * args.frac_lazy))
    n_self = N - n_champ - n_league - n_lazy
    env_mode = np.array([0] * n_self + [1] * n_champ + [2] * n_league + [3] * n_lazy)
    lazy = LazyOpponents(env, np.random.default_rng(args.seed + 1))
    for e in np.where(env_mode == 3)[0]:
        lazy.reset_env(e)
    learner_red = np.array([e % 2 == 0 for e in range(N)])
    env_opp = [league.sample() if env_mode[e] == 2 else None for e in range(N)]
    env_gen = np.full(N, league.champion_gen)

    def learner_agents():
        """(env, agent) pairs controlled by the learner, in a fixed slot order."""
        idx = []
        for e in range(N):
            if env_mode[e] == 0:
                idx += [e * P + k for k in range(P)]
            else:
                base = 0 if learner_red[e] else ts
                idx += [e * P + base + k for k in range(ts)]
        return np.array(idx)

    def opponent_groups():
        groups = collections.defaultdict(list)
        for e in range(N):
            if env_mode[e] in (0, 3):  # self-play, or lazy (scripted, handled separately)
                continue
            base = ts if learner_red[e] else 0
            if env_mode[e] == 1:
                key = "champ"
            else:  # league; plays the champion until the league has model members
                key = env_opp[e] if env_opp[e] is not None else "champ"
            groups[key] += [e * P + base + k for k in range(ts)]
        return {k: np.array(v) for k, v in groups.items()}

    def lazy_agents():
        """(env indices, agent indices) of the lazy opponents."""
        envs, agents = [], []
        for e in np.where(env_mode == 3)[0]:
            base = ts if learner_red[e] else 0
            envs += [e] * ts
            agents += [base + k for k in range(ts)]
        return np.array(envs, np.int64), np.array(agents, np.int64)

    learn_idx = learner_agents()
    lazy_envs, lazy_ags = lazy_agents()
    groups = opponent_groups()
    L = len(learn_idx)
    T = args.rollout
    mix = ", ".join(f"{k} {lbl}" for k, lbl in ((n_self, "self-play"), (n_league, "vs league (PFSP)"),
                                               (n_champ, "vs champion"), (n_lazy, "vs lazy")) if k)
    print(f"{ts}v{ts} on {args.stadium}: {N} games in parallel ({mix})"
          + (f", ping {args.ping_ms:.0f}ms" if args.ping_ms else "") + f" -> {run_dir}")

    b_obs = torch.zeros(T, L, D)
    b_act = torch.zeros(T, L, dtype=torch.long)
    b_logp = torch.zeros(T, L)
    b_val = torch.zeros(T, L)
    b_rew = np.zeros((T, L), np.float32)
    b_done = np.zeros((T, L), np.float32)

    obs = env.reset().reshape(N * P, D)
    st = dict(champ=champ_results, league=collections.deque(maxlen=300),
              goals=collections.deque(maxlen=300), contact=collections.deque(maxlen=300), len=collections.deque(maxlen=300),
              passes=collections.deque(maxlen=300), corner=collections.deque(maxlen=300),
              idle=collections.deque(maxlen=300))
    last_view = games // args.view_every
    last_promo = games // args.promote_every
    last_snap = games // args.snapshot_every_games
    last_ckpt = games // args.ckpt_every_games
    viewer_proc = None
    t0, sps_t, sps_n = time.time(), time.time(), 0
    ent_hist, vl_hist, kl_hist = [], [], []
    log_path = os.path.join(run_dir, "log.csv")

    def save(path):
        torch.save(dict(model=snap(), opt=opt.state_dict(), games=games, updates=updates,
                        model_kwargs=mkw, champ_results=list(champ_results),
                        env_kwargs=dict(team_size=ts, stadium=args.stadium, ping_ms=args.ping_ms,
                                        frame_skip=args.frame_skip)), path)

    def launch_viewer():
        nonlocal viewer_proc
        if args.no_view or (viewer_proc is not None and viewer_proc.is_alive()):
            return
        from .viewer import run_viewer
        payload = dict(team_size=ts, stadium=args.stadium, ping_ms=args.ping_ms, model_kwargs=mkw,
                       learner=snap(), champion=league.champion_state, rows=rows[-400:], seconds=args.view_seconds, skip_to_s=args.view_skip,
                       corner_box=args.corner_box, corner_hold=args.corner_hold,
                       idle=dict(idle_dist=args.idle_dist, idle_hold_s=args.idle_hold, idle_penalty=args.idle_penalty),
                       games=games, title=f"{ts}v{ts} {args.stadium} | {args.name} | {games} games")
        viewer_proc = mp.get_context("spawn").Process(target=run_viewer, args=(payload,), daemon=True)
        viewer_proc.start()

    try:
        while games < args.total_games:
            w_speed, w_touch, w_btg, spirit, stage, fade = reward_weights(args, games)
            env.set_reward_weights(w_speed, w_touch, w_btg)
            env.set_idle_scale(ramp(games, args.idle_start_games, args.idle_ramp_games))
            env.set_corner_scale(ramp(games, args.corner_start_games, args.corner_ramp_games))
            env.set_team_rewards(args.pass_reward, args.crowd_penalty, spirit, args.pass_min_dist, args.crowd_dist,
                                 args.pass_backward)
            ent_coef = args.ent_coef + (args.ent_coef_final - args.ent_coef) * min(1.0, games / args.ent_anneal_games)
            for t in range(T):
                o = torch.from_numpy(obs)
                o_l = o[learn_idx]
                a, logp, v = model.act(o_l.to(dev, non_blocking=True), nm, no)
                a, logp, v = a.cpu(), logp.cpu(), v.cpu()
                acts = np.zeros(N * P, np.int64)
                acts[learn_idx] = a.numpy()
                for key, idx in groups.items():
                    net = league.champion_model() if key == "champ" else league.model(key)
                    acts[idx] = net.act(o[idx].to(opp_dev, non_blocking=True), nm, no)[0].cpu().numpy()
                if len(lazy_envs):
                    acts[lazy_envs * P + lazy_ags] = lazy.actions(lazy_envs, lazy_ags)
                b_obs[t], b_act[t], b_logp[t], b_val[t] = o_l, a, logp, v
                nobs, rew, done, finished = env.step(acts.reshape(N, P))
                obs = nobs.reshape(N * P, D)
                b_rew[t] = rew.reshape(-1)[learn_idx]
                b_done[t] = np.repeat(done, P)[learn_idx]
                regroup = False
                for (e, rg, bg, ticks, rc, bc, npass, cpen, ipen) in finished:
                    games += 1
                    mode = env_mode[e]
                    if mode == 0:
                        st["goals"].append(rg + bg)
                        st["contact"].append((rc + bc) / env.P / 60.0)
                        st["passes"].append(npass)
                        st["len"].append(ticks / 60.0)
                        st["corner"].append(cpen / 60.0)
                        st["idle"].append(ipen / 60.0)
                    elif mode == 3:  # lazy opponent: training only, nothing to record
                        lazy.reset_env(e)
                        learner_red[e] = not learner_red[e]
                        regroup = True
                    else:
                        mine, theirs = (rg, bg) if learner_red[e] else (bg, rg)
                        res = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
                        if mode == 1:
                            if env_gen[e] == league.champion_gen:
                                champ_results.append(res)
                            env_gen[e] = league.champion_gen
                        else:
                            if env_opp[e] is not None:
                                league.record(env_opp[e], res)
                                st["league"].append(res)
                            env_opp[e] = league.sample()
                        learner_red[e] = not learner_red[e]
                        regroup = True
                if regroup:
                    learn_idx = learner_agents()
                    groups = opponent_groups()
                    lazy_envs, lazy_ags = lazy_agents()
            sps_n += T * L

            # ---- GAE
            with torch.no_grad():
                _, v_last = model(torch.from_numpy(obs)[learn_idx].to(dev), nm, no)
                v_last = v_last.cpu()
            vals = b_val.numpy()
            adv = np.zeros((T, L), np.float32)
            last = np.zeros(L, np.float32)
            nv = v_last.numpy()
            for t in reversed(range(T)):
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + args.gamma * nv * nonterm - vals[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
                nv = vals[t]
            ret = torch.from_numpy(adv + vals).reshape(-1)
            advt = torch.from_numpy(adv).reshape(-1)
            advt = (advt - advt.mean()) / (advt.std() + 1e-8)
            fo, fa, fl = b_obs.reshape(T * L, D).to(dev), b_act.reshape(-1).to(dev), b_logp.reshape(-1).to(dev)
            ret, advt = ret.to(dev), advt.to(dev)

            # ---- PPO
            model.train()
            n = T * L
            mb = n // args.minibatches
            kls, ents, vls = [], [], []
            stop = False
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=dev)
                for s in range(0, n - mb + 1, mb):
                    i = perm[s:s + mb]
                    lp, ent, v = model.evaluate(fo[i], fa[i], nm, no)
                    ratio = (lp - fl[i]).exp()
                    pa = advt[i]
                    pl = -torch.min(ratio * pa, ratio.clamp(1 - args.clip, 1 + args.clip) * pa).mean()
                    vl = 0.5 * (v - ret[i]).pow(2).mean()
                    loss = pl + args.vf_coef * vl - ent_coef * ent.mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                    opt.step()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - (lp - fl[i])).mean().item())
                    ents.append(ent.mean().item())
                    vls.append(vl.item())
                    if kls[-1] > args.target_kl:
                        stop = True
                        break
                if stop:
                    break
            model.eval()
            updates += 1
            ent_hist.append(fmean(ents))
            vl_hist.append(fmean(vls))
            kl_hist.append(fmean(kls))

            # ---- league bookkeeping
            if games // args.promote_every > last_promo:
                last_promo = games // args.promote_every
                if args.promote_eval_games > 0:
                    te = time.time()
                    res = eval_vs(model, league.champion_model(), env_kwargs, args.promote_eval_games,
                                  seed=random.randrange(1 << 30), dev=dev, opp_dev=opp_dev)
                    champ_results.clear()
                    champ_results.extend(res)
                    eval_note = (f"won {res.count(1.0)} drew {res.count(0.5)} lost {res.count(0.0)} of {len(res)}, "
                                 f"{time.time() - te:.0f} s")
                else:
                    eval_note = f"over {len(champ_results)} training games"
                wc = fmean(champ_results)
                la = league.league_winrate()
                ok_league = la is None or la >= args.promote_min_league
                if len(champ_results) >= args.promote_min_games and wc >= args.promote_threshold and ok_league:
                    old = league.champion_gen
                    league.promote(snap(), games)
                    champ_results.clear()
                    print(f"*** PROMOTED at {games} games: now champion gen {league.champion_gen} "
                          f"(scored {wc:.0%} vs gen {old}: {eval_note})", flush=True)
                else:
                    print(f"--- {games} games: {wc:.0%} vs champion gen {league.champion_gen} "
                          f"(needs {args.promote_threshold:.0%}; {eval_note})"
                          + ("" if ok_league else f", but league avg {la:.2f} is under {args.promote_min_league:.2f}"),
                          flush=True)
                league.save()
            if games // args.snapshot_every_games > last_snap:
                last_snap = games // args.snapshot_every_games
                league.add(snap(), "snapshot", games)
                league.save()
            if refresh_opponents(env_opp, league):  # a promotion or snapshot may have evicted a member
                groups = opponent_groups()
            if games // args.ckpt_every_games > last_ckpt:
                last_ckpt = games // args.ckpt_every_games
                save(os.path.join(run_dir, f"ckpt_{games:07d}.pt"))
            if updates % 25 == 0:
                save(os.path.join(run_dir, "latest.pt"))
                league.save()

            if updates % args.log_every == 0:
                now = time.time()
                sps = sps_n / (now - sps_t)
                sps_t, sps_n = now, 0
                row = dict(games=games, updates=updates, sps=sps, champion_gen=league.champion_gen,
                           win_champ=fmean(champ_results), win_league=fmean(st["league"]),
                           goals_per_game=fmean(st["goals"]), contact_s=fmean(st["contact"]), passes_per_game=fmean(st["passes"]),
                           game_len_s=fmean(st["len"]), corner_pen_s=fmean(st["corner"]),
                           idle_pen_s=fmean(st["idle"]), entropy=fmean(ent_hist[-args.log_every:]),
                           value_loss=fmean(vl_hist[-args.log_every:]), approx_kl=fmean(kl_hist[-args.log_every:]),
                           reward_stage=stage, push_reward_fade=fade, minutes=(now - t0) / 60)
                rows.append(row)
                header = None
                if os.path.exists(log_path):
                    with open(log_path, newline="") as f:
                        header = next(csv.reader(f), None)
                if header != list(row):
                    # new file, or a resumed run from before a column was added: rewrite it with
                    # the current columns (missing old values become 0) so the CSV stays aligned
                    with open(log_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(row), restval=0.0, extrasaction="ignore")
                        w.writeheader()
                        w.writerows(rows)
                else:
                    with open(log_path, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=list(row)).writerow(row)
                print(f"games {games:7d} | {sps / 1000:5.1f}k sps | gen {league.champion_gen:3d} | "
                      + (f"win vs champ {row['win_champ']:.2f} | " if args.frac_champ else "")
                      + f"league {row['win_league']:.2f} | "
                      f"goals/g {row['goals_per_game']:5.2f} contact {row['contact_s']:5.1f}s/g"
                      + (f" corner {row['corner_pen_s']:4.1f}s/g" if args.corner_box else "")
                      + (f" idle {row['idle_pen_s']:4.1f}s/g" if args.idle_penalty else "")
                      + (f" passes/g {row['passes_per_game']:4.1f}" if args.pass_reward else "") + " | "
                      f"ent {row['entropy']:.2f} vl {row['value_loss']:.4f} kl {row['approx_kl']:.4f}", flush=True)

            if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
                print(f"--- time limit of {args.max_minutes:g} min reached at {games} games, stopping", flush=True)
                break
            if games // args.view_every > last_view:
                last_view = games // args.view_every
                launch_viewer()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        save(os.path.join(run_dir, "latest.pt"))
        league.save()
        print(f"saved {os.path.abspath(os.path.join(run_dir, 'latest.pt'))} ({games} games, champion gen "
              f"{league.champion_gen})")


if __name__ == "__main__":
    main()
