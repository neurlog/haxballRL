"""Build the app's data files from the trained models (run from the repo root):

    python app/tools/build_assets.py

* app/policies/<mode>.json  - the bot network weights + the mode's game settings (from export_policy)
* app/assets/header.svg     - the app's animated header (same clip as the README banner)
* .github/header.svg        - the README banner: logo + title over a kickoff-to-kickoff 3v3 bot match
* app/assets/icon.png       - the logo: a red HaxBall-style player with "RL", transparent background
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "headless-bot"))

from export_policy import export  # noqa: E402

from hax.env import HaxEnv, stadium_scales  # noqa: E402
from hax.model import load_policy  # noqa: E402

APP = os.path.join(ROOT, "app")
MODES = {
    "1v1-classic": dict(title="1v1 Classic", ckpt="1v1-classic/models/main-200kv9.pt", team_size=1,
                        stadium="classic", stadium_name="Classic", games=200000,
                        blurb="Champion after 200,000 games of league self-play (run v9, gen 15). You vs the bot on the Classic map."),
    "3v3-big": dict(title="3v3 Big", ckpt="3v3-big/models/champion_5k.pt", team_size=3, stadium="big",
                    stadium_name="Big", games=5000,
                    blurb="Champion after 5,000 games. 3 bots vs your team on the Big map; empty spots on your team are filled with bots."),
}


def build_policies():
    index = []
    for key, m in MODES.items():
        pol = export(os.path.join(ROOT, m["ckpt"]))
        env = HaxEnv(1, team_size=m["team_size"], stadium=m["stadium"])
        ps, ds = stadium_scales(m["stadium"])
        pol["mode"] = dict(key=key, title=m["title"], team_size=m["team_size"], stadium=m["stadium_name"],
                           goal_x=env.goal_x, pos_scale=ps.tolist(), dist_scale=ds, frame_skip=env.frame_skip,
                           time_limit_min=3, games_trained=m["games"], blurb=m["blurb"])
        with open(os.path.join(APP, "policies", f"{key}.json"), "w") as f:
            json.dump(pol, f)
        index.append(pol["mode"])
        print("policy", key, "->", m["ckpt"])
    with open(os.path.join(APP, "policies", "index.json"), "w") as f:
        json.dump(index, f, indent=1)


def build_icon(size=1024):
    """A HaxBall player disc with an "RL" avatar, drawn the way HaxBall draws avatars:
    white Arial Black (weight 900) with a black drop shadow, on the red team colour with a
    black outline. Transparent background."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c, r = size // 2, int(size * 0.497)  # as large as the canvas allows (was 0.47)
    pygame.draw.circle(s, (0, 0, 0, 255), (c, c), r)                       # black outline
    pygame.draw.circle(s, (229, 110, 86, 255), (c, c), int(r * 0.92))      # HaxBall red (#E56E56)
    font_path = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
    font = pygame.font.Font(font_path, int(size * 0.425)) if os.path.exists(font_path) else \
        pygame.font.SysFont("arialblack,arial", int(size * 0.425), bold=True)
    txt = font.render("RL", True, (255, 255, 255))
    shadow = font.render("RL", True, (0, 0, 0))
    x, y = c - txt.get_width() // 2, c - txt.get_height() // 2
    off = int(size * 0.012)  # HaxBall offsets the black copy by ~7/70 of the font size
    s.blit(shadow, (x + off, y + off))
    s.blit(txt, (x, y))
    pygame.image.save(s, os.path.join(APP, "assets", "icon.png"))
    pygame.image.save(s, os.path.join(APP, "build", "icon.png"))  # Windows icon: the round player, no background
    _make_icns(os.path.join(APP, "assets", "icon.png"), os.path.join(APP, "build", "icon.icns"))  # macOS icon
    print("icon", size, "(+ macOS icon.icns, Windows icon.png)")


def _make_icns(png, out):
    """1024px PNG -> .icns with all the sizes macOS wants (needs macOS's sips + iconutil)."""
    import shutil
    import subprocess
    import tempfile
    d = os.path.join(tempfile.mkdtemp(), "icon.iconset")
    os.makedirs(d)
    for px in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            name = f"icon_{px}x{px}{'@2x' if scale == 2 else ''}.png"
            subprocess.run(["sips", "-z", str(px * scale), str(px * scale), png, "--out", os.path.join(d, name)],
                           check=True, capture_output=True)
    subprocess.run(["iconutil", "-c", "icns", d, "-o", out], check=True)
    shutil.rmtree(os.path.dirname(d))


def record_kickoff_loop(min_s=14, max_s=30, fps=30, seed=3):
    """Frames of the 3v3 champion playing itself from one kickoff to the next (a goal, the
    celebration, then the reset), so the clip loops seamlessly: it starts and ends on the
    kickoff spots. Returns a list of [ball_x, ball_y, p0x, p0y, ...] at `fps`."""
    net = load_policy(os.path.join(ROOT, MODES["3v3-big"]["ckpt"]))
    rng_seed = seed
    while True:
        torch.manual_seed(rng_seed)  # the bots sample their actions: seed it so the clip is reproducible
        env = HaxEnv(1, team_size=3, stadium="big", seed=rng_seed)
        obs = env.observe()
        frames, recording, was_waiting = [], False, True
        for _ in range(30 * 60 * 3):  # up to 3 minutes of decisions (30/s)
            st = env.state
            waiting = bool(env.kickoff_waiting()[0])
            if waiting and not was_waiting:  # a new kickoff just appeared (after a goal)
                if recording and min_s * fps <= len(frames) <= max_s * fps:
                    return frames
                recording, frames = True, []
            if recording:
                frames.append([round(float(v), 1) for v in np.concatenate([st.ball_pos[0], st.player_pos[0].ravel()])])
                if len(frames) > max_s * fps:
                    recording, frames = False, []
            was_waiting = waiting
            a = net.act(torch.from_numpy(obs.reshape(-1, env.obs_dim)), env.n_mates, env.n_opps)[0].numpy()
            obs, _, done, _ = env.step(a.reshape(1, -1))
            if done[0]:
                break
        rng_seed += 1


def build_readme_header(fps=15, out=".github/header.svg", radius=12, height=150):
    """Animated SVG banner for the GitHub README: the 3v3 champion playing itself on the Big
    pitch (SMIL animation from the recorded replay, plays in any browser via <img>), with the
    logo and title on top. Wide and short: the pitch is shown at full width and a "camera"
    pans up and down to follow the ball. Pure vector, no fonts or images to load."""
    teams = ["red"] * 3 + ["blue"] * 3
    frames = record_kickoff_loop()[:: 30 // fps]
    W, H = 1000, height
    sc, ox, oy = W / 1180, W / 2, H / 2
    X = lambda x: round(ox + x * sc, 1)  # noqa: E731
    Y = lambda y: round(oy + y * sc, 1)  # noqa: E731
    dur = f"{len(frames) / fps:.2f}s"
    bw, bh, gy, ko = 550, 240, 80, 80
    # camera: smoothed ball height, clamped so the view never leaves the pitch
    half_view = H / 2 / sc
    lim = bh + 20 - half_view
    cam, cams = frames[0][1], []
    for f in frames:
        cam += 0.15 * (f[1] - cam)
        cams.append(max(-lim, min(lim, cam)))
    cam_vals = ";".join(f"0 {round(-c * sc, 1)}" for c in cams)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="haxballRL">',
             '<defs><radialGradient id="fade" cx=".5" cy=".5" r=".6" gradientTransform="translate(0 0) scale(1 1)">'
             '<stop offset="0" stop-color="#000" stop-opacity=".5"/><stop offset=".6" stop-color="#000" stop-opacity=".22"/>'
             '<stop offset="1" stop-color="#000" stop-opacity=".1"/></radialGradient>'
             f'<clipPath id="r"><rect width="{W}" height="{H}" rx="{radius}"/></clipPath></defs>',
             '<g clip-path="url(#r)">', f'<rect width="{W}" height="{H}" fill="#5f7d4b"/>',
             f'<g><animateTransform attributeName="transform" type="translate" values="{cam_vals}" dur="{dur}" repeatCount="indefinite"/>']
    n = 14
    sw = 2 * bw / n
    for i in range(1, n, 2):
        parts.append(f'<rect x="{X(-bw + i * sw)}" y="{Y(-bh - 40)}" width="{round(sw * sc, 1)}" height="{round((2 * bh + 80) * sc, 1)}" fill="#58744a"/>')
    line = 'fill="none" stroke="#c7e6bd" stroke-width="3"'
    parts.append(f'<rect x="{X(-bw)}" y="{Y(-bh)}" width="{round(2 * bw * sc, 1)}" height="{round(2 * bh * sc, 1)}" {line}/>')
    parts.append(f'<line x1="{X(0)}" y1="{Y(-bh)}" x2="{X(0)}" y2="{Y(bh)}" stroke="#c7e6bd" stroke-width="3"/>')
    parts.append(f'<circle cx="{X(0)}" cy="{Y(0)}" r="{round(ko * sc, 1)}" {line}/>')
    for gx, col, d, post in ((-bw, "#e56e56", -1, "#ffcccc"), (bw, "#5689e5", 1, "#ccccff")):
        parts.append(f'<path d="M{X(gx)} {Y(-gy)} L{X(gx + d * 30)} {Y(-gy + 20)} L{X(gx + d * 30)} {Y(gy - 20)} L{X(gx)} {Y(gy)}" '
                     'fill="none" stroke="#111" stroke-width="3"/>')
        parts.append(f'<line x1="{X(gx)}" y1="{Y(-gy)}" x2="{X(gx)}" y2="{Y(gy)}" stroke="{col}" stroke-width="3"/>')
        for py in (-gy, gy):
            parts.append(f'<circle cx="{X(gx)}" cy="{Y(py)}" r="{round(8 * sc, 1)}" fill="{post}" stroke="#000" stroke-width="2"/>')

    def disc(idx, r, fill):
        xs = ";".join(str(X(f[idx])) for f in frames)
        ys = ";".join(str(Y(f[idx + 1])) for f in frames)
        return (f'<circle cx="{X(frames[0][idx])}" cy="{Y(frames[0][idx + 1])}" r="{round(r * sc, 1)}" fill="{fill}" stroke="#000" stroke-width="2">'
                f'<animate attributeName="cx" values="{xs}" dur="{dur}" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{ys}" dur="{dur}" repeatCount="indefinite"/></circle>')
    for k, team in enumerate(teams):
        parts.append(disc(2 + 2 * k, 15, "#e56e56" if team == "red" else "#5689e5"))
    parts.append(disc(0, 10, "#f5f5f5"))
    parts.append("</g>")
    parts.append(f'<rect width="{W}" height="{H}" fill="url(#fade)"/>')
    # logo (a HaxBall player disc with an "RL" avatar) + title, centred together
    R, gap, title_w = 40, 18, 230  # title_w: approx. width of "haxballRL" at 44px
    total = 2 * R + gap + title_w
    cx, cy = (W - total) / 2 + R, H / 2
    tx = cx + R + gap
    font = "'Arial Black','Arial Bold',Gadget,sans-serif"
    ui = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    parts += [f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="#000"/>',
              f'<circle cx="{cx}" cy="{cy}" r="{R * 0.9}" fill="#e56e56"/>',
              f'<text x="{cx + 2}" y="{cy + 2}" font-family="{font}" font-weight="900" font-size="32" text-anchor="middle" dominant-baseline="central" fill="#000">RL</text>',
              f'<text x="{cx}" y="{cy}" font-family="{font}" font-weight="900" font-size="32" text-anchor="middle" dominant-baseline="central" fill="#fff">RL</text>',
              f'<text x="{tx}" y="{cy + 4}" font-family="{ui}" font-weight="800" font-size="44" fill="#fff" textLength="{title_w}" lengthAdjust="spacingAndGlyphs">haxballRL</text>',
              f'<text x="{tx + 2}" y="{cy + 32}" font-family="{ui}" font-size="17" fill="#eef3ee">Play against RL-trained bots</text>',
              '</g></svg>']
    path = os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print("readme header", out, f"{os.path.getsize(path) / 1024:.0f} KB, {len(frames)} frames, {dur} loop, {W}x{H}")


if __name__ == "__main__":
    only = sys.argv[1:]
    if not only or "policies" in only:
        build_policies()
    if not only or "icon" in only:
        build_icon()
    if not only or "header" in only:
        build_readme_header()                                            # GitHub README banner
        build_readme_header(out="app/assets/header.svg", radius=0, height=210)  # the app's header: edge to edge, a bit taller
