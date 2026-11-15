// Draws the header background: the 3v3 champion playing itself on the Big stadium,
// replayed from assets/replay.json (recorded with the Python simulator).
"use strict";
(function () {
  const cv = document.getElementById("pitch");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const BG_W = 550, BG_H = 240, GOAL_Y = 80, KO = 80;  // Big stadium
  const sc = W / 1180, ox = W / 2, oy = H / 2 + 10;
  const X = (x) => ox + x * sc, Y = (y) => oy + y * sc;

  function field() {
    ctx.fillStyle = "#5f7d4b"; ctx.fillRect(0, 0, W, H);
    const n = 14, sw = (2 * BG_W) / n;
    ctx.fillStyle = "#58744a";
    for (let i = 1; i < n; i += 2) ctx.fillRect(X(-BG_W + i * sw), Y(-BG_H), sw * sc + 1, 2 * BG_H * sc);
    ctx.strokeStyle = "#c7e6bd"; ctx.lineWidth = 3;
    ctx.strokeRect(X(-BG_W), Y(-BG_H), 2 * BG_W * sc, 2 * BG_H * sc);
    ctx.beginPath(); ctx.moveTo(X(0), Y(-BG_H)); ctx.lineTo(X(0), Y(BG_H)); ctx.stroke();
    ctx.beginPath(); ctx.arc(X(0), Y(0), KO * sc, 0, Math.PI * 2); ctx.stroke();
    for (const [gx, col, dir] of [[-BG_W, "#e56e56", -1], [BG_W, "#5689e5", 1]]) {
      ctx.strokeStyle = "#111"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(X(gx), Y(-GOAL_Y)); ctx.lineTo(X(gx + dir * 30), Y(-GOAL_Y + 20));
      ctx.lineTo(X(gx + dir * 30), Y(GOAL_Y - 20)); ctx.lineTo(X(gx), Y(GOAL_Y)); ctx.stroke();
      ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(X(gx), Y(-GOAL_Y)); ctx.lineTo(X(gx), Y(GOAL_Y)); ctx.stroke();
      for (const py of [-GOAL_Y, GOAL_Y]) disc(gx, py, 8, dir < 0 ? "#ffcccc" : "#ccccff");
    }
  }
  function disc(x, y, r, fill) {
    ctx.beginPath(); ctx.arc(X(x), Y(y), r * sc + 2, 0, Math.PI * 2); ctx.fillStyle = "#000"; ctx.fill();
    ctx.beginPath(); ctx.arc(X(x), Y(y), r * sc, 0, Math.PI * 2); ctx.fillStyle = fill; ctx.fill();
  }

  window.haxballrl.replay().then((rep) => {
    let i = 0;
    const step = () => {
      const f = rep.frames[i];
      field();
      for (let k = 0; k < rep.teams.length; k++) disc(f[2 + 2 * k], f[3 + 2 * k], 15, rep.teams[k] === "red" ? "#e56e56" : "#5689e5");
      disc(f[0], f[1], 10, "#f5f5f5");
      i = (i + 1) % rep.frames.length;
    };
    step();
    setInterval(step, 1000 / rep.fps);
  });
})();
