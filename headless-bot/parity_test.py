"""Check that policy.js builds the same observation and logits as the Python code.

    python headless-bot/parity_test.py [checkpoint]

Plays a few hundred decisions of random-action games in HaxEnv, records the raw state the
Node bot would see in a real room, runs policy.js on it, and compares.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from export_policy import export  # noqa: E402

from hax.env import RED, HaxEnv  # noqa: E402
from hax.model import HaxNet, load_policy  # noqa: E402


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt:
        net = load_policy(ckpt)
    else:
        from hax.env import ENT_DIM, SELF_DIM
        net = HaxNet(SELF_DIM, ENT_DIM)
        for p in net.parameters():  # random but non-trivial weights (mate encoder included)
            torch.nn.init.normal_(p, 0, 0.3)
        torch.save(net.state_dict(), ckpt := os.path.join(tempfile.mkdtemp(), "rand.pt"))
    pol = export(ckpt)
    env = HaxEnv(4, team_size=1, seed=7)
    samples = []
    for step in range(900):
        if step % 10 == 0:  # random actions held for a third of a second, so players cover the pitch
            a = env.rng.integers(0, 18, (env.n, env.P))
        obs, _, _, _ = env.step(a)
        if step % 7:
            continue
        st = env.state
        for e in range(env.n):
            ps = [dict(x=float(st.player_pos[e, k, 0]), y=float(st.player_pos[e, k, 1]),
                       vx=float(st.player_vel[e, k, 0]), vy=float(st.player_vel[e, k, 1])) for k in range(env.P)]
            ball = dict(x=float(st.ball_pos[e, 0]), y=float(st.ball_pos[e, 1]),
                        vx=float(st.ball_vel[e, 0]), vy=float(st.ball_vel[e, 1]))
            play_state = 2 if env.celeb[e] > 0 else (0 if env.kickoff_waiting()[e] else 1)
            for k in range(env.P):
                team = 1 if env.team[e, k] == RED else 2
                mates = [ps[j] for j in env.ent_order[k][: env.n_mates]]
                opps = [ps[j] for j in env.ent_order[k][env.n_mates:]]
                my, their = (0, 1) if team == 1 else (1, 0)
                g = dict(team=team, playState=play_state, kickingTeam=1 if env.kicking[e] == RED else 2,
                         timeS=float(env.play_ticks[e] / 60), timeLimitS=env.time_limit / 60,
                         myScore=int(env.score[e, my]), theirScore=int(env.score[e, their]),
                         lastActions=[int(x) for x in env.last_act[e, k]], goalX=env.goal_x)
                with torch.no_grad():
                    lg = net(torch.from_numpy(obs[e, k][None]), env.n_mates, env.n_opps)[0][0].numpy()
                samples.append(dict(me=ps[k], mates=mates, opps=opps, ball=ball, g=g,
                                    obs=obs[e, k].tolist(), logits=lg.tolist()))
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "pol.json"), "w") as f:
        json.dump(pol, f)
    with open(os.path.join(tmp, "samples.json"), "w") as f:
        json.dump(samples, f)
    js = f"""
const P = require({json.dumps(os.path.join(tmp, 'pol.json'))});
const S = require({json.dumps(os.path.join(tmp, 'samples.json'))});
const pol = require({json.dumps(os.path.join(HERE, 'policy.js'))});
let mo = 0, ml = 0, agree = 0;
for (const s of S) {{
  const o = pol.buildObs(s.me, s.mates, s.opps, s.ball, s.g);
  if (o.length !== s.obs.length) throw new Error(`obs length ${{o.length}} != ${{s.obs.length}}`);
  for (let i = 0; i < o.length; i++) mo = Math.max(mo, Math.abs(o[i] - s.obs[i]));
  const l = pol.logits(P, o, s.mates.length, s.opps.length);
  for (let i = 0; i < l.length; i++) ml = Math.max(ml, Math.abs(l[i] - s.logits[i]));
  if (pol.argmax(l) === pol.argmax(s.logits)) agree++;
}}
console.log(JSON.stringify({{n: S.length, max_obs_diff: mo, max_logit_diff: ml, argmax_agree: agree / S.length}}));
"""
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    ok = r["max_obs_diff"] < 1e-5 and r["max_logit_diff"] < 1e-3
    print(f"{r['n']} samples | max obs diff {r['max_obs_diff']:.2e} | max logit diff {r['max_logit_diff']:.2e} | "
          f"argmax agreement {r['argmax_agree']:.1%} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
