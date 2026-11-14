"""PPO with an AlphaStar-style league on HaxballGym physics.

Each env slot has a fixed role, so the number of learner agents never changes:

    self    both teams are the learner (two learner agents per game)
    champ   learner vs the frozen champion (supplies the promotion statistics)
    league  learner vs a PFSP-sampled league member (resampled every game)
    bot     learner vs the scripted bot (off by default; --frac-bot)

Default mix: 45% self-play, 30% champion, 25% league. The scripted bot is used only as an
evaluation yardstick (`--eval-bot-every`), not as a training opponent.

The learner swaps colour every game. Every `promote_every` games the learner is promoted
to champion if it beat the champion >= 75% over >= 60 games and averages >= 50% against
the league's past champions and snapshots.

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

from .bots import scripted_actions
from .env import ENT_DIM, N_ACTIONS, SELF_DIM, HaxEnv
from .league import BOT_ID, League
from .model import HaxNet, load_policy

MODES = ("self", "champ", "league", "bot")


@torch.no_grad()
def eval_vs_bot(model, n_games, args):
    """Learner vs the scripted bot, one game per env (half as red, half as blue).
    Returns the mean result (1 win, 0.5 draw, 0 loss). Not used for training."""
    from .env import HaxEnv as _Env
    env = _Env(n_games, team_size=args.team_size, stadium=args.stadium, seed=random.randrange(1 << 30),
               ping_ms=args.ping_ms, jitter_ms=args.jitter_ms, frame_skip=args.frame_skip,
               time_limit_s=args.time_limit, overtime_limit_s=args.overtime_limit,
               kickoff_timeout_s=args.kickoff_timeout)
    ts, P = env.ts, env.P
    learner_red = np.arange(n_games) % 2 == 0
    mask = np.zeros((n_games, P), bool)
    mask[learner_red, :ts] = True
    mask[~learner_red, ts:] = True
    obs, res = env.observe(), {}
    while len(res) < n_games:
        a = scripted_actions(env)
        o = torch.from_numpy(obs[mask])
        a[mask] = model.act(o, env.n_mates, env.n_opps)[0].numpy()
        obs, _, _, fin = env.step(a)
        for e, rg, bg, *_ in fin:
            if e not in res:
                mine, theirs = (rg, bg) if learner_red[e] else (bg, rg)
                res[e] = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return float(np.mean(list(res.values())))


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--name", default="main", help="run name; saved to <mode folder>/runs/<name>")
    a("--runs-dir", default="runs")
    a("--resume", action="store_true")
    a("--init-from", default=None, help="start from these weights (e.g. a 1v1 checkpoint for 3v3)")
    a("--total-games", type=int, default=10**9)
    # game
    a("--team-size", type=int, default=1)
    a("--stadium", default="classic")
    a("--ping-ms", type=float, default=40.0)
    a("--jitter-ms", type=float, default=4.0)
    a("--frame-skip", type=int, default=2, help="physics ticks per decision (2 = 30 decisions/s)")
    a("--time-limit", type=float, default=180.0)
    a("--overtime-limit", type=float, default=60.0)
    a("--kickoff-timeout", type=float, default=10.0, help="training-only: walk the kicking player into the ball if a kickoff stalls this long (s); 0 = off")
    # reward
    a("--goal-reward", type=float, default=1.0, help="+ for each goal scored, - for each conceded")
    a("--win-reward", type=float, default=3.0, help="+ for winning the game, - for losing it, 0 for a draw")
    # RLGym-style dense rewards, per decision (30/s), each term in [0,1] or [-1,1]. Weights move
    # linearly from the 'early' to the 'late' value over --reward-stage-games, like the RLGym
    # guide's stages: first learn to reach and touch the ball, then shift to moving it to goal.
    # Nothing is removed completely. Set all to 0 for goals + wins only.
    a("--speed-early", type=float, default=0.02, help="speed towards the ball (clipped at 0)")
    a("--speed-late", type=float, default=0.002)
    a("--touch-early", type=float, default=0.2, help="touch, scaled by how hard the ball was hit")
    a("--touch-late", type=float, default=0.01)
    a("--btg-early", type=float, default=0.01, help="ball velocity towards the goal you attack")
    a("--btg-late", type=float, default=0.01)
    a("--reward-stage-games", type=float, default=3000)
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
    a("--ent-coef-final", type=float, default=0.01, help="RLGym guide: ~0.01, don't cut it while still training")
    a("--ent-anneal-games", type=float, default=10000)
    a("--max-grad", type=float, default=0.5)
    a("--target-kl", type=float, default=0.03)
    a("--hidden", type=int, default=256)
    a("--ent-hidden", type=int, default=128)
    a("--threads", type=int, default=4)
    a("--seed", type=int, default=0)
    # league
    a("--frac-champ", type=float, default=0.30)
    a("--frac-league", type=float, default=0.25)
    a("--frac-bot", type=float, default=0.0)
    a("--bot-in-league", action="store_true", help="also let PFSP pick the scripted bot")
    a("--eval-bot-every", type=int, default=1000, help="eval games vs the scripted bot every N games (0 = off)")
    a("--eval-bot-games", type=int, default=64)
    a("--promote-every", type=int, default=500)
    a("--promote-threshold", type=float, default=0.75)
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
    a("--no-view", action="store_true")
    a("--log-every", type=int, default=10)
    a("--ckpt-every-games", type=int, default=1000)
    if defaults:
        p.set_defaults(**defaults)
    return p


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
    env = HaxEnv(N, team_size=ts, stadium=args.stadium, seed=args.seed, ping_ms=args.ping_ms,
                 jitter_ms=args.jitter_ms, frame_skip=args.frame_skip, time_limit_s=args.time_limit,
                 overtime_limit_s=args.overtime_limit, kickoff_timeout_s=args.kickoff_timeout,
                 goal_reward=args.goal_reward, win_reward=args.win_reward)
    P, D = env.P, env.obs_dim
    nm, no = env.n_mates, env.n_opps
    mkw = dict(self_dim=SELF_DIM, ent_dim=ENT_DIM, n_actions=N_ACTIONS, hidden=args.hidden, ent_hidden=args.ent_hidden)
    model = HaxNet(**mkw)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    ctor = lambda: HaxNet(**mkw)  # noqa: E731
    league = League(run_dir, ctor, topk=args.pfsp_topk, explore=args.pfsp_explore, power=args.pfsp_power,
                    max_size=args.league_size, include_bot=args.bot_in_league)
    snap = lambda: {k: v.detach().clone() for k, v in model.state_dict().items()}  # noqa: E731

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
    else:
        if args.init_from:
            src = load_policy(args.init_from)
            model.load_state_dict(src.state_dict())
            print(f"initialised from {args.init_from}")
        league.set_champion(snap(), 0, 0)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=1)

    # ---- env slot roles (fixed), learner colour flips every game
    n_champ = int(round(N * args.frac_champ))
    n_league = int(round(N * args.frac_league))
    n_bot = int(round(N * args.frac_bot))
    n_self = N - n_champ - n_league - n_bot
    env_mode = np.array([0] * n_self + [1] * n_champ + [2] * n_league + [3] * n_bot)
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
            if env_mode[e] == 0:
                continue
            base = ts if learner_red[e] else 0
            if env_mode[e] == 1:
                key = "champ"
            elif env_mode[e] == 3:
                key = BOT_ID
            else:  # league; plays the champion until the league has model members
                key = env_opp[e] if env_opp[e] is not None else "champ"
            groups[key] += [e * P + base + k for k in range(ts)]
        return {k: np.array(v) for k, v in groups.items()}

    learn_idx = learner_agents()
    groups = opponent_groups()
    L = len(learn_idx)
    T = args.rollout
    print(f"{ts}v{ts} on {args.stadium}: {N} games in parallel ({n_self} self-play, {n_champ} vs champion, "
          f"{n_league} vs league, {n_bot} vs bot), {L} learner agents, ping {args.ping_ms:.0f}ms, "
          f"{60 // args.frame_skip} decisions/s -> {run_dir}")

    b_obs = torch.zeros(T, L, D)
    b_act = torch.zeros(T, L, dtype=torch.long)
    b_logp = torch.zeros(T, L)
    b_val = torch.zeros(T, L)
    b_rew = np.zeros((T, L), np.float32)
    b_done = np.zeros((T, L), np.float32)

    obs = env.reset().reshape(N * P, D)
    st = dict(champ=champ_results, league=collections.deque(maxlen=300), bot=collections.deque(maxlen=200),
              goals=collections.deque(maxlen=300), contact=collections.deque(maxlen=300), len=collections.deque(maxlen=300))
    last_view = games // args.view_every
    last_eval = games // args.eval_bot_every if args.eval_bot_every else 0
    bot_eval = rows[-1].get("win_bot", float("nan")) if rows else float("nan")
    last_promo = games // args.promote_every
    last_snap = games // args.snapshot_every_games
    last_ckpt = games // args.ckpt_every_games
    viewer_proc = None
    t0, sps_t, sps_n = time.time(), time.time(), 0
    ent_hist, vl_hist, kl_hist = [], [], []
    log_path = os.path.join(run_dir, "log.csv")

    def save(path):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), games=games, updates=updates,
                        model_kwargs=mkw, champ_results=list(champ_results),
                        env_kwargs=dict(team_size=ts, stadium=args.stadium, ping_ms=args.ping_ms,
                                        frame_skip=args.frame_skip)), path)

    def launch_viewer():
        nonlocal viewer_proc
        if args.no_view or (viewer_proc is not None and viewer_proc.is_alive()):
            return
        from .viewer import run_viewer
        payload = dict(team_size=ts, stadium=args.stadium, ping_ms=args.ping_ms, model_kwargs=mkw,
                       learner=snap(), champion=league.champion_state, rows=rows[-400:], seconds=args.view_seconds,
                       games=games, title=f"{ts}v{ts} {args.stadium} | {args.name} | {games} games")
        viewer_proc = mp.get_context("spawn").Process(target=run_viewer, args=(payload,), daemon=True)
        viewer_proc.start()

    try:
        while games < args.total_games:
            stage = min(1.0, games / args.reward_stage_games) if args.reward_stage_games > 0 else 1.0
            lerp = lambda a, b: a + (b - a) * stage  # noqa: E731
            env.set_reward_weights(lerp(args.speed_early, args.speed_late), lerp(args.touch_early, args.touch_late),
                                   lerp(args.btg_early, args.btg_late))
            ent_coef = args.ent_coef + (args.ent_coef_final - args.ent_coef) * min(1.0, games / args.ent_anneal_games)
            for t in range(T):
                o = torch.from_numpy(obs)
                a, logp, v = model.act(o[learn_idx], nm, no)
                acts = np.zeros(N * P, np.int64)
                acts[learn_idx] = a.numpy()
                for key, idx in groups.items():
                    if key == BOT_ID:
                        acts[idx] = scripted_actions(env).reshape(-1)[idx]
                    else:
                        net = league.champion_model() if key == "champ" else league.model(key)
                        acts[idx] = net.act(o[idx], nm, no)[0].numpy()
                b_obs[t], b_act[t], b_logp[t], b_val[t] = o[learn_idx], a, logp, v
                nobs, rew, done, finished = env.step(acts.reshape(N, P))
                obs = nobs.reshape(N * P, D)
                b_rew[t] = rew.reshape(-1)[learn_idx]
                b_done[t] = np.repeat(done, P)[learn_idx]
                regroup = False
                for (e, rg, bg, ticks, rc, bc) in finished:
                    games += 1
                    mode = env_mode[e]
                    if mode == 0:
                        st["goals"].append(rg + bg)
                        st["contact"].append((rc + bc) / 2 / 60.0)
                        st["len"].append(ticks / 60.0)
                    else:
                        mine, theirs = (rg, bg) if learner_red[e] else (bg, rg)
                        res = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
                        if mode == 1:
                            if env_gen[e] == league.champion_gen:
                                champ_results.append(res)
                            env_gen[e] = league.champion_gen
                        elif mode == 2:
                            if env_opp[e] is not None:
                                league.record(env_opp[e], res)
                                st["league"].append(res)
                            env_opp[e] = league.sample()
                        else:
                            st["bot"].append(res)
                            league.record(BOT_ID, res)
                        learner_red[e] = not learner_red[e]
                        regroup = True
                if regroup:
                    learn_idx = learner_agents()
                    groups = opponent_groups()
            sps_n += T * L

            # ---- GAE
            with torch.no_grad():
                _, v_last = model(torch.from_numpy(obs)[learn_idx], nm, no)
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
            fo, fa, fl = b_obs.reshape(T * L, D), b_act.reshape(-1), b_logp.reshape(-1)

            # ---- PPO
            model.train()
            n = T * L
            mb = n // args.minibatches
            kls, ents, vls = [], [], []
            stop = False
            for _ in range(args.epochs):
                perm = torch.randperm(n)
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
                wc = fmean(champ_results)
                la = league.league_winrate()
                ok_league = la is None or la >= args.promote_min_league
                if len(champ_results) >= args.promote_min_games and wc >= args.promote_threshold and ok_league:
                    old = league.champion_gen
                    league.promote(snap(), games)
                    champ_results.clear()
                    print(f"*** PROMOTED at {games} games: learner is champion gen {league.champion_gen} "
                          f"(won {wc:.0%} vs gen {old}, league avg {'n/a' if la is None else f'{la:.2f}'}) "
                          f"| league now {len(league.members)} members")
                else:
                    print(f"--- promotion check at {games} games: {wc:.0%} vs champion gen {league.champion_gen} over "
                          f"{len(champ_results)} games (need {args.promote_threshold:.0%} over "
                          f"{args.promote_min_games}+), league avg {'n/a' if la is None else f'{la:.2f}'} "
                          f"| hardest: {league.summary()}")
                league.save()
            if games // args.snapshot_every_games > last_snap:
                last_snap = games // args.snapshot_every_games
                league.add(snap(), "snapshot", games)
                league.save()
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
                           win_champ=fmean(champ_results), win_league=fmean(st["league"]), win_bot=fmean(st["bot"]) if args.frac_bot > 0 else bot_eval,
                           goals_per_game=fmean(st["goals"]), contact_s=fmean(st["contact"]),
                           game_len_s=fmean(st["len"]), entropy=fmean(ent_hist[-args.log_every:]),
                           value_loss=fmean(vl_hist[-args.log_every:]), approx_kl=fmean(kl_hist[-args.log_every:]),
                           reward_stage=stage, minutes=(now - t0) / 60)
                rows.append(row)
                new = not os.path.exists(log_path)
                with open(log_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"games {games:7d} | upd {updates:6d} | {sps / 1000:5.1f}k sps | gen {league.champion_gen:3d} | "
                      f"win champ {row['win_champ']:.2f} league {row['win_league']:.2f} bot {row['win_bot']:.2f} | "
                      f"goals/g {row['goals_per_game']:5.2f} contact {row['contact_s']:5.1f}s/g | "
                      f"ent {row['entropy']:.2f} vl {row['value_loss']:.4f} kl {row['approx_kl']:.4f}", flush=True)

            if args.eval_bot_every and games // args.eval_bot_every > last_eval:
                last_eval = games // args.eval_bot_every
                te = time.time()
                bot_eval = eval_vs_bot(model, args.eval_bot_games, args)
                print(f"--- eval at {games} games: {bot_eval:.0%} vs scripted bot over {args.eval_bot_games} games "
                      f"({time.time() - te:.0f}s, not used for training)", flush=True)
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
