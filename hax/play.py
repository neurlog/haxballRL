"""Watch or play games in real time.

Each side is one of:
    latest      the run's latest learner          champion   the run's current champion
    gen3        champion generation 3 (any N)     ckpt:5000  the checkpoint saved at ~5000 games
    human       you (arrows/WASD move, Space/X kick)
    <path.pt>   any checkpoint or league .pt file

Examples (from the repo root):
    python 1v1-classic/play.py --red gen2 --blue gen1        # two champions
    python 1v1-classic/play.py --red human --blue champion   # you vs the champion
    python 1v1-classic/play.py --red latest --blue gen1 --games 64 --fast  # quick eval, no window
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch

from .env import HaxEnv, keys_to_action
from .model import load_policy


def resolve(spec: str, run_dir: str):
    if spec == "human":
        return spec
    if spec == "latest":
        return os.path.join(run_dir, "latest.pt")
    if spec == "champion":
        return os.path.join(run_dir, "league", "champion.pt")
    if spec.startswith("gen") and spec[3:].isdigit():
        return os.path.join(run_dir, "league", f"champion_gen{int(spec[3:]):03d}.pt")
    if spec.startswith("ckpt:"):
        want = int(spec[5:])
        cks = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
        if not cks:
            raise SystemExit(f"no checkpoints in {run_dir}")
        return min(cks, key=lambda p: abs(int(os.path.basename(p)[5:12]) - want))
    return spec


def human_action(pg, keys):
    dx = int(keys[pg.K_RIGHT] or keys[pg.K_d]) - int(keys[pg.K_LEFT] or keys[pg.K_a])
    dy = int(keys[pg.K_DOWN] or keys[pg.K_s]) - int(keys[pg.K_UP] or keys[pg.K_w])
    return keys_to_action(dx, dy, bool(keys[pg.K_SPACE] or keys[pg.K_x]))


def main(argv=None, mode_dir=".", team_size=1, stadium="classic"):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="main", help="run name under runs/")
    p.add_argument("--red", default="latest")
    p.add_argument("--blue", default="champion")
    p.add_argument("--ping-ms", type=float, default=0.0)
    p.add_argument("--deterministic", action="store_true", help="argmax actions instead of sampling")
    p.add_argument("--games", type=int, default=0, help="stop after N games (0 = until Esc)")
    p.add_argument("--fast", action="store_true", help="no window: play --games games in parallel and print results")
    args = p.parse_args(argv)
    run_dir = os.path.join(mode_dir, "runs", args.run)
    sides = []
    for name in (args.red, args.blue):
        r = resolve(name, run_dir)
        if r == "human":
            sides.append((name, r))
        else:
            if not os.path.exists(r):
                raise SystemExit(f"{name}: {r} not found")
            sides.append((name, load_policy(r)))
    ts = team_size

    def actions(env, obs):
        o = torch.from_numpy(obs.reshape(-1, env.obs_dim))
        a = np.zeros((env.n, env.P), np.int64)
        for s, (_, pol) in enumerate(sides):
            sl = slice(s * ts, (s + 1) * ts)
            if pol == "human":
                a[:, sl] = 0
            else:
                rows = np.concatenate([np.arange(e * env.P, e * env.P + env.P)[sl] for e in range(env.n)])
                a[:, sl] = pol.act(o[rows], env.n_mates, env.n_opps, args.deterministic)[0].numpy().reshape(env.n, ts)
        return a

    if args.fast:
        n = max(1, args.games or 32)
        env = HaxEnv(n, team_size=ts, stadium=stadium, seed=int(time.time()), ping_ms=args.ping_ms)
        obs, res = env.observe(), []
        done_env = np.zeros(n, bool)
        while not done_env.all():
            obs, _, _, fin = env.step(actions(env, obs))
            for e, rg, bg, *_ in fin:
                if not done_env[e]:
                    done_env[e] = True
                    res.append((rg, bg))
        r = np.array(res)
        w, d = (r[:, 0] > r[:, 1]).mean(), (r[:, 0] == r[:, 1]).mean()
        print(f"red {args.red} vs blue {args.blue}: {len(r)} games | red wins {w:.0%} draws {d:.0%} "
              f"losses {1 - w - d:.0%} | goals {r[:, 0].mean():.2f} - {r[:, 1].mean():.2f}")
        return

    import pygame

    from .viewer import Renderer

    env = HaxEnv(1, team_size=ts, stadium=stadium, seed=int(time.time()), ping_ms=args.ping_ms)
    rnd = Renderer(stadium, title=f"HaxBall RL: {args.red} vs {args.blue}")
    obs = env.observe()
    clock = pygame.time.Clock()
    label = f"red = {args.red}   blue = {args.blue}   ping {args.ping_ms:.0f}ms   (Esc quits)"
    played, tally = 0, [0, 0, 0]
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
        a = actions(env, obs)
        keys = pygame.key.get_pressed()
        for s, (_, pol) in enumerate(sides):
            if pol == "human":
                ha = human_action(pygame, keys)
                a[0, s * ts] = ha if s == 0 else _mirror(ha)
        obs, _, _, fin = env.step(a)
        f = env.frame(0)
        if fin:
            _, rg, bg, *_ = fin[0]
            f["score"] = [rg, bg]
            tally[0 if rg > bg else (2 if bg > rg else 1)] += 1
            played += 1
            print(f"game {played}: {rg}-{bg}   (red W/D/L so far {tally[0]}/{tally[1]}/{tally[2]})")
        rnd.draw_frame(f, label)
        pygame.display.flip()
        if fin:
            pygame.time.wait(1500)
            if args.games and played >= args.games:
                pygame.quit()
                return
        clock.tick(60 // env.frame_skip)


def _mirror(a: int) -> int:
    """Human input is in world coordinates; blue's actions are given in its mirrored frame."""
    from .env import DIRS
    dx, dy = DIRS[a % 9]
    return keys_to_action(-dx, dy, a >= 9)


if __name__ == "__main__":
    main()
