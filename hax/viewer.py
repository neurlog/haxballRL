"""pygame rendering: the pitch, a live game, and training charts.

`run_viewer(payload)` is what the trainer launches in a separate process every N games:
the current learner (red) plays the current champion (blue) in real time for a few
seconds next to charts of the training log.
Keys in the window: D = sampled/argmax actions, B = blue: champion <-> learner, Esc = close.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

GRASS, GRASS2, LINE = (105, 135, 85), (98, 127, 79), (199, 230, 189)
RED_C, BLUE_C, BALL_C, TXT = (229, 110, 86), (86, 137, 229), (245, 245, 245), (235, 235, 235)
SCALE = 1.0


def _arc_points(a, b, curve_deg, n=12):
    a, b = np.array(a, float), np.array(b, float)
    th = math.radians(abs(curve_deg))
    chord = b - a
    L = np.linalg.norm(chord)
    if th < 1e-6 or L < 1e-6:
        return [tuple(a), tuple(b)]
    nrm = np.array([-chord[1], chord[0]]) / L
    mid = (a + b) / 2
    off = (L / 2) / math.tan(th / 2)
    c1, c2 = mid + nrm * off, mid - nrm * off
    c = c1 if np.linalg.norm(c1) < np.linalg.norm(c2) else c2  # nets bulge away from the pitch
    r = np.linalg.norm(a - c)
    a0, a1 = math.atan2(*(a - c)[::-1]), math.atan2(*(b - c)[::-1])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return [tuple(c + r * np.array([math.cos(a0 + da * t / n), math.sin(a0 + da * t / n)])) for t in range(n + 1)]


class Renderer:
    def __init__(self, stadium="classic", width=None, height=480, title="HaxBall RL", panel_width=0):
        import pygame
        from haxballgym import stadium_text

        self.pg = pygame
        d = json.loads(stadium_text(stadium))
        self.st = d
        self.bg = d.get("bg", {})
        self.fw, self.fh = d.get("width", 420), d.get("height", 200)
        self.field_w = width or int(2 * self.fw * SCALE + 20)
        self.field_h = height
        pygame.init()
        self.screen = pygame.display.set_mode((self.field_w + panel_width, self.field_h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("menlo", 16)
        self.big = pygame.font.SysFont("menlo", 22, bold=True)
        self.small = pygame.font.SysFont("menlo", 12)
        vx = d["vertexes"]
        self.nets = []
        for s in d["segments"]:
            if s.get("vis", True) is False:
                continue
            a, b = (vx[s["v0"]]["x"], vx[s["v0"]]["y"]), (vx[s["v1"]]["x"], vx[s["v1"]]["y"])
            self.nets.append(_arc_points(a, b, s.get("curve", 0.0)))
        self.posts = [x for x in d.get("discs", []) if "pos" in x]
        self.goals = d.get("goals", [])

    def w2s(self, x, y):
        return (int(self.field_w / 2 + x * SCALE), int(self.field_h / 2 + 20 + y * SCALE))

    def draw_field(self):
        pg, s = self.pg, self.screen
        s.fill((60, 70, 60))
        bw, bh = self.bg.get("width", 370), self.bg.get("height", 170)
        x0, y0 = self.w2s(-self.fw, -self.fh)
        pg.draw.rect(s, GRASS, (x0, y0, int(2 * self.fw * SCALE), int(2 * self.fh * SCALE)))
        for i in range(-6, 7, 2):
            sx0, sy0 = self.w2s(i * 60, -bh)
            pg.draw.rect(s, GRASS2, (sx0, sy0, int(60 * SCALE), int(2 * bh * SCALE)))
        pg.draw.rect(s, LINE, (*self.w2s(-bw, -bh), int(2 * bw * SCALE), int(2 * bh * SCALE)), 3)
        pg.draw.line(s, LINE, self.w2s(0, -bh), self.w2s(0, bh), 3)
        pg.draw.circle(s, LINE, self.w2s(0, 0), int(self.bg.get("kickOffRadius", 75) * SCALE), 3)
        for g in self.goals:
            col = RED_C if g["team"] == "red" else BLUE_C
            pg.draw.line(s, col, self.w2s(*g["p0"]), self.w2s(*g["p1"]), 3)
        for pts in self.nets:
            pg.draw.lines(s, (20, 20, 20), False, [self.w2s(*p) for p in pts], 2)
        for p in self.posts:
            pg.draw.circle(s, (20, 20, 20), self.w2s(*p["pos"]), int(p.get("radius", 8) * SCALE) + 1)
            col = tuple(int(p.get("color", "FFFFFF")[i:i + 2], 16) for i in (0, 2, 4))
            pg.draw.circle(s, col, self.w2s(*p["pos"]), int(p.get("radius", 8) * SCALE) - 1)

    def draw_frame(self, f, label=""):
        pg, s = self.pg, self.screen
        self.draw_field()
        for (x, y), team, kick in zip(f["players"], f["teams"], f["kicking"]):
            c = self.w2s(x, y)
            pg.draw.circle(s, (255, 255, 255) if kick else (0, 0, 0), c, int(15 * SCALE) + 2)
            pg.draw.circle(s, RED_C if team == 2 else BLUE_C, c, int(15 * SCALE))
        pg.draw.circle(s, (0, 0, 0), self.w2s(*f["ball"]), int(10 * SCALE) + 2)
        pg.draw.circle(s, BALL_C, self.w2s(*f["ball"]), int(10 * SCALE))
        t = f["time_s"]
        clock = f"{int(t // 60)}:{int(t % 60):02d}" + (" OT" if f["overtime"] else "")
        txt = self.big.render(f"{f['score'][0]}  -  {f['score'][1]}     {clock}", True, TXT)
        s.blit(txt, (self.field_w / 2 - txt.get_width() / 2, 8))
        if label:
            lt = self.font.render(label, True, TXT)
            s.blit(lt, (self.field_w / 2 - lt.get_width() / 2, self.field_h - 24))

    def draw_chart(self, rect, title, series, ymin=None, ymax=None):
        """series: list of (label, color, values)."""
        pg, s = self.pg, self.screen
        x, y, w, h = rect
        pg.draw.rect(s, (35, 38, 35), rect)
        s.blit(self.small.render(title, True, TXT), (x + 4, y + 2))
        vals = [v for _, _, vs in series for v in vs if v is not None and np.isfinite(v)]
        if not vals:
            return
        lo = min(vals) if ymin is None else ymin
        hi = max(vals) if ymax is None else ymax
        if hi - lo < 1e-9:
            hi = lo + 1.0
        py0, ph = y + 18, h - 24
        n = max(len(vs) for _, _, vs in series)
        for i, (lab, col, vs) in enumerate(series):
            pts = [(x + 4 + (w - 8) * (j + n - len(vs)) / max(1, n - 1), py0 + ph * (1 - (v - lo) / (hi - lo)))
                   for j, v in enumerate(vs) if v is not None and np.isfinite(v)]
            if len(pts) > 1:
                pg.draw.lines(s, col, False, pts, 2)
            s.blit(self.small.render(lab, True, col), (x + w - 70, y + 2 + 12 * i))
        s.blit(self.small.render(f"{hi:.3g}", True, (150, 150, 150)), (x + 4, py0))
        s.blit(self.small.render(f"{lo:.3g}", True, (150, 150, 150)), (x + 4, py0 + ph - 12))

    def draw_panel(self, rows, x0):
        col = lambda k: [r.get(k) for r in rows]  # noqa: E731
        w, h, pad = 250, 108, 8
        charts = [
            ("win rate", [("champ", (240, 200, 80), col("win_champ")), ("league", (120, 200, 240), col("win_league"))],
             0, 1),
            ("champion generation", [("gen", (240, 200, 80), col("champion_gen"))], None, None),
            ("goals / game (self-play)", [("goals", (240, 240, 240), col("goals_per_game"))], 0, None),
            ("ball contact s / game", [("contact", (240, 240, 240), col("contact_s"))], 0, None),
            ("entropy", [("ent", (160, 230, 160), col("entropy"))], None, None),
            ("value loss", [("vl", (230, 160, 160), col("value_loss"))], 0, None),
            ("approx KL", [("kl", (230, 200, 160), col("approx_kl"))], 0, None),
            ("game length (s, self-play)", [("len", (180, 180, 230), col("game_len_s"))], 0, None),
        ]
        for i, (t, ser, lo, hi) in enumerate(charts):
            cx, cy = x0 + pad + (i % 2) * (w + pad), pad + (i // 2) * (h + pad)
            self.draw_chart((cx, cy, w, h), t, ser, lo, hi)


def run_viewer(payload):
    """Entry point of the viewer process."""
    import pygame
    import torch

    from .env import HaxEnv
    from .model import HaxNet

    torch.set_num_threads(1)
    ts = payload["team_size"]
    env = HaxEnv(1, team_size=ts, stadium=payload.get("stadium", "classic"),
                 seed=int(time.time()), ping_ms=payload.get("ping_ms", 0.0))
    nets = {}
    for k in ("learner", "champion"):
        if payload.get(k) is not None:
            n = HaxNet(**payload["model_kwargs"])
            n.load_state_dict(payload[k])
            n.eval()
            nets[k] = n
    rows = payload.get("rows", [])
    panel = 2 * 250 + 3 * 8
    r = Renderer(payload.get("stadium", "classic"), title=payload.get("title", "HaxBall RL training"), panel_width=panel)
    r.draw_panel(rows, r.field_w)
    blue_modes = ["champion", "learner"]
    bm, det = 0, False
    obs = env.observe()
    clock = pygame.time.Clock()
    t_end = time.time() + payload.get("seconds", 20.0)
    while time.time() < t_end:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_d:
                det = not det
            if e.type == pygame.KEYDOWN and e.key == pygame.K_b:
                bm = (bm + 1) % len(blue_modes)
        o = torch.from_numpy(obs.reshape(-1, env.obs_dim))
        acts = nets["learner"].act(o, env.n_mates, env.n_opps, det)[0].numpy().reshape(1, -1)
        mode = blue_modes[bm]
        net = nets.get(mode, nets["learner"])
        acts[:, ts:] = net.act(o[ts:], env.n_mates, env.n_opps, det)[0].numpy()
        obs, _, _, fin = env.step(acts)
        f = env.frame(0)
        if fin:
            _, rg, bg, *_ = fin[0]
            f["score"] = [rg, bg]
        r.draw_frame(f, f"red = learner  vs  blue = {mode}   [{'argmax' if det else 'sampled'}]  "
                        f"games {payload.get('games', 0)}   (B: switch blue, D: argmax, Esc: close)")
        r.draw_panel(rows, r.field_w)
        pygame.display.flip()
        clock.tick(60 // env.frame_skip)
    pygame.quit()
