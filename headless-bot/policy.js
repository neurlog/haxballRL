// Observation builder + policy forward pass, in plain JavaScript.
// Mirrors hax/env.py (HaxEnv.observe) and hax/model.py (HaxNet) exactly; the parity test
// (python headless-bot/parity_test.py) checks both against the Python code.
"use strict";

const RED = 1, BLUE = 2; // node-haxball team ids
const DIRS = [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
const PX = 1 / 420, PY = 1 / 200, VS = 1 / 5;

function actionKeys(a) { // joint action -> [dx, dy, kick] in the agent's own (mirrored) frame
  const d = DIRS[a % 9];
  return [d[0], d[1], a >= 9 ? 1 : 0];
}

/**
 * Build one agent's observation.
 *  me, mates[], opps[]: {x, y, vx, vy} in world coordinates
 *  ball: {x, y, vx, vy}
 *  g: {team (1 red / 2 blue), playState (0 kickoff, 1 playing, 2 goal), kickingTeam,
 *      timeS, timeLimitS, myScore, theirScore, lastActions: [a_t-1, a_t-2], goalX}
 */
function buildObs(me, mates, opps, ball, g) {
  const sx = g.team === RED ? 1 : -1;
  const gx = g.goalX;
  const px = sx * me.x, py = me.y, vx = sx * me.vx, vy = me.vy;
  const bx = sx * ball.x, by = ball.y, bvx = sx * ball.vx, bvy = ball.vy;
  const rbx = bx - px, rby = by - py;
  const o = [
    px * PX, py * PY, vx * VS, vy * VS,
    rbx * PX, rby * PY, bvx * VS, bvy * VS,
    bx * PX, by * PY,
    (gx - px) * PX, (0 - py) * PY,
    (-gx - px) * PX, (0 - py) * PY,
    Math.hypot(rbx, rby) / 400,
  ];
  for (const a of g.lastActions) o.push(...actionKeys(a));
  const waiting = g.playState === 0 ? 1 : 0;
  o.push(waiting, waiting && g.team === g.kickingTeam ? 1 : 0, g.playState === 2 ? 1 : 0);
  o.push(Math.min(g.timeS / g.timeLimitS, 1), g.timeS >= g.timeLimitS ? 1 : 0);
  o.push(Math.tanh((g.myScore - g.theirScore) / 2));
  for (const p of [...mates, ...opps]) {
    const qx = sx * p.x, qy = p.y;
    const rx = qx - px, ry = qy - py;
    o.push(rx * PX, ry * PY, sx * p.vx * VS, p.vy * VS, (bx - qx) * PX, (by - qy) * PY,
      Math.hypot(rx, ry) / 400, Math.hypot(bx - qx, by - qy) / 400);
  }
  return o;
}

// ---------------------------------------------------------------- network
function linear(x, L) {
  const out = new Array(L.b.length);
  for (let i = 0; i < out.length; i++) {
    let s = L.b[i];
    const w = L.w[i];
    for (let j = 0; j < x.length; j++) s += w[j] * x[j];
    out[i] = s;
  }
  return out;
}
const relu = (v) => v.map((z) => (z > 0 ? z : 0));
const mlp = (x, layers) => layers.reduce((h, L) => relu(linear(h, L)), x);

function pool(rows, enc, width) {
  if (rows.length === 0) return new Array(2 * width).fill(0);
  const hs = rows.map((r) => mlp(r, enc));
  const mean = new Array(width).fill(0), max = new Array(width).fill(-Infinity);
  for (const h of hs) for (let i = 0; i < width; i++) { mean[i] += h[i] / hs.length; if (h[i] > max[i]) max[i] = h[i]; }
  return mean.concat(max);
}

/** Policy logits (18) for one observation. P = parsed policy.json. */
function logits(P, obs, nMates, nOpps) {
  const sd = P.self_dim, ed = P.ent_dim;
  const s = obs.slice(0, sd);
  const ents = [];
  for (let k = 0; k < nMates + nOpps; k++) ents.push(obs.slice(sd + k * ed, sd + (k + 1) * ed));
  const h = mlp(s, P.self_enc)
    .concat(pool(ents.slice(0, nMates), P.mate_enc, P.ent_hidden))
    .concat(pool(ents.slice(nMates), P.opp_enc, P.ent_hidden));
  return linear(mlp(h, P.trunk), P.pi);
}

function argmax(v) { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b; }
function sample(v, temperature = 1) {
  const m = Math.max(...v);
  const e = v.map((z) => Math.exp((z - m) / temperature));
  const tot = e.reduce((a, b) => a + b, 0);
  let r = Math.random() * tot;
  for (let i = 0; i < e.length; i++) { r -= e[i]; if (r <= 0) return i; }
  return e.length - 1;
}

module.exports = { RED, BLUE, DIRS, actionKeys, buildObs, logits, argmax, sample };
