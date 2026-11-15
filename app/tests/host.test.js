// Bot-seating and decision test with a fake node-haxball room (no network needed):
//   node tests/host.test.js
"use strict";
const assert = require("assert");
const path = require("path");
const fs = require("fs");
const host = require("../room/host");

// the app's copy of the policy code must match the parity-tested one in headless-bot/
assert.strictEqual(fs.readFileSync(path.join(__dirname, "..", "room", "policy.js"), "utf8"),
                   fs.readFileSync(path.join(__dirname, "..", "..", "headless-bot", "policy.js"), "utf8"),
                   "app/room/policy.js differs from headless-bot/policy.js");

function fakeRoom() {
  const players = [];
  const inputs = [];
  const disc = (x, y) => ({ pos: { x, y }, speed: { x: 0, y: 0 } });
  const r = {
    players, inputs,
    getPlayer: (id) => players.find((p) => p.id === id),
    fakePlayerJoin: (id, name) => players.push({ id, name, team: { id: 0 }, disc: null }),
    fakePlayerLeave: (id) => players.splice(players.findIndex((p) => p.id === id), 1),
    setPlayerTeam: (id, team) => { const p = r.getPlayer(id); p.team = { id: team }; p.disc = team ? disc(team === 1 ? -400 : 400, players.length * 30 - 60) : null; },
    fakeSendPlayerInput: (key, id) => inputs.push({ key, id }),
    admins: [], started: false,
    setPlayerAdmin: (id, v) => { if (v) r.admins.push(id); },
    sendAnnouncement: () => {}, kickPlayer: () => {}, startGame: () => { r.started = true; }, stopGame: () => {},
    gameState: { state: 1, timeElapsed: 10, redScore: 0, blueScore: 0, goalConcedingTeam: { id: 1 },
                 physicsState: { discs: [disc(0, 0)] } },
  };
  return r;
}

for (const mode of ["1v1-classic", "3v3-big"]) {
  const P = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "policies", `${mode}.json`), "utf8"));
  const ts = P.mode.team_size;
  const room = fakeRoom();
  host.bots.clear();
  host._attachForTest(room, P, { deterministic: false, maxPlayers: 4 }, true);
  host.setupLobby();
  const team = (t) => room.players.filter((p) => p.team.id === t);
  const botCount = () => host.bots.size;
  assert.strictEqual(botCount(), 1, "the room starts with one bot");
  assert.strictEqual(team(1).length, 1, "...on red");
  assert.ok(!room.started, "nothing starts by itself");

  // Add bot up to 2 x team size: red fills first, then blue, then spectators; extra clicks do nothing
  for (let i = 0; i < 10; i++) host.requestAddBot();
  assert.strictEqual(botCount(), 2 * ts, `add bot stops at ${2 * ts}`);
  assert.strictEqual(team(1).length, ts, "red full of bots");
  assert.strictEqual(team(2).length, ts, "then blue");
  // Remove bot: newest first; can go down to zero, extra clicks do nothing
  for (let i = 0; i < 10; i++) host.requestRemoveBot();
  assert.strictEqual(botCount(), 0, "remove bot goes down to zero");
  host.requestAddBot();
  for (let i = 1; i < 2 * ts; i++) host.requestAddBot();
  assert.strictEqual(botCount(), 2 * ts, "and back up");

  // the room maker joins first -> owner (admin), spectating; a friend joins -> not admin
  const join = (id, name) => { const p = { id, name, team: { id: 0 }, disc: null }; room.players.push(p); host.onJoin(p); return p; };
  const me = join(1, "me"), friend = join(2, "friend");
  assert.strictEqual(host.owner(), 1, "first person to join owns the room");
  assert.deepStrictEqual(room.admins, [1], "only the owner is admin");
  assert.strictEqual(me.team.id, 0, "people join as spectators");
  assert.ok(!room.started, "joining doesn't start the game");

  // owner drags a blue bot to spectators and the friend into blue -> that bot stops playing
  const blueBot = team(2).find((p) => host.bots.has(p.id));
  room.setPlayerTeam(blueBot.id, 0);
  room.setPlayerTeam(friend.id, 2);
  for (let t = 0; t < 30; t++) host.decide();
  const senders = new Set(room.inputs.map((i) => i.id));
  assert.ok(!senders.has(blueBot.id), "a bot moved to spectators sits out");
  const playingBots = [...host.bots.keys()].filter((id) => room.getPlayer(id).team.id !== 0);
  assert.ok(playingBots.every((id) => senders.has(id)), "every bot on a team sends inputs");
  assert.ok(room.inputs.every((i) => Number.isInteger(i.key) && i.key >= 0 && i.key < 32), "inputs are valid key states");

  // owner leaves -> the friend becomes owner
  room.players.splice(room.players.indexOf(me), 1);
  host.onLeave(me);
  assert.strictEqual(host.owner(), 2, "ownership passes on");
  assert.ok(room.admins.includes(2), "new owner is admin");
  console.log(`PASS ${mode}: starts with 1 bot, add/remove between 0 and ${2 * ts}, owner = first joiner, bots in spectators idle, ${playingBots.length} playing bots sent inputs`);
}

// 60 s auto-close (shortened to 0.3 s here): nobody joins -> room closes; someone joins -> timer cancelled
(async () => {
  const P = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "policies", "1v1-classic.json"), "utf8"));
  for (const someoneJoins of [false, true]) {
    const room = fakeRoom(); let left = false; room.leave = () => { left = true; };
    const events = [];
    host.bots.clear();
    host._attachForTest(room, P, { maxPlayers: 4 }, true, events);
    host.setupLobby();
    host.startJoinTimer(0.3);
    if (someoneJoins) { const p = { id: 1, name: "me", team: { id: 0 }, disc: null }; room.players.push(p); host.onJoin(p); }
    await new Promise((r) => setTimeout(r, 500));
    const types = events.map((e) => e.type);
    assert.ok(types.includes("countdown"), "the app is told to show the countdown");
    if (someoneJoins) {
      assert.ok(types.includes("joined") && !types.includes("closed") && !left, "a join cancels the auto-close");
      console.log("PASS auto-close: someone joined in time -> countdown hidden, room stays open");
    } else {
      const closed = events.find((e) => e.type === "closed");
      assert.ok(closed && left && /nobody joined/.test(closed.reason), "nobody joined -> room closed");
      console.log(`PASS auto-close: nobody joined -> room closed (${closed.reason.replace("0 s", "0.3 s")})`);
    }
  }
})().catch((e) => { console.error(e); process.exit(1); });
