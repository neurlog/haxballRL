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
  assert.strictEqual(team(1).length, ts, "lobby: bots fill red");
  assert.strictEqual(team(2).length, ts, "lobby: bots fill blue");
  assert.ok(!room.started, "nothing starts by itself");

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
  console.log(`PASS ${mode}: lobby ${ts}v${ts} bots, owner = first joiner, bots in spectators idle, ${playingBots.length} playing bots sent inputs`);
}
