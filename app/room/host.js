// Hosts a private HaxBall room with trained haxballrl bots. Runs in its own process
// (Electron utilityProcess): the app sends {type: "start", ...} and gets events back.
//
// Lobby: the room opens with one bot (on red) and nothing starts by itself. The app's
// "Add bot" / "Remove bot" buttons add or remove bots, up to 2 x team_size (2 in 1v1,
// 6 in 3v3). A new bot takes the first free team spot (red, then blue), else spectators. The first person to join (whoever made the room) becomes the
// owner (admin): they drag bots and friends between red / blue / spectators in HaxBall's
// lobby and press Start. Everyone else joins as a spectator. If the owner leaves, the next
// person in the room becomes owner. Bots on a team play; bots in spectators idle.
// If nobody joins within 60 s of the room opening, it closes itself.
//
// Observation + network: ../../headless-bot/policy.js logic, copied into ./policy.js so the
// packaged app is self-contained (parity-tested against the Python trainer).
"use strict";
const { Room, Utils, OperationType } = require("node-haxball")();
const pol = require("./policy");

const RED = 1, BLUE = 2, SPEC = 0;
const FIRST_BOT_ID = 65535;

let room = null;
let P = null; // policy (weights + mode)
let cfg = null;
const bots = new Map(); // id -> {team, lastActions, lastKey}
let tick = 0;

let send = (msg) => { if (process.parentPort) process.parentPort.postMessage(msg); else console.log(JSON.stringify(msg)); };
const log = (text) => send({ type: "log", text });
let after = (fn) => Utils.runAfterGameTick(fn);
const humans = () => (room ? room.players.filter((p) => !bots.has(p.id)) : []);
const discInfo = (d) => ({ x: d.pos.x, y: d.pos.y, vx: d.speed.x, vy: d.speed.y });

function playerList() {
  if (!room) return [];
  return room.players.map((p) => ({ id: p.id, name: p.name, team: p.team ? p.team.id : 0, bot: bots.has(p.id) }));
}
const pushPlayers = () => send({ type: "players", players: playerList() });

// ------------------------------------------------------------------ bots
function nextBotId() {
  let id = FIRST_BOT_ID;
  while (bots.has(id)) id--;
  return id;
}

function addBot(team) {
  const id = nextBotId();
  bots.set(id, { team, lastActions: [0, 0], lastKey: -1 });
  room.fakePlayerJoin(id, "bot", "", "", "bot-conn", "bot-auth");  // plain "bot": no flag, no avatar
  after(() => room.setPlayerTeam(id, team));
}

let owner = null; // player id of the room owner (admin)
const JOIN_TIMEOUT_S = 60; // close the room if nobody joins within this long after it opens
let joinTimer = null;
let closeReason = null; // set when we close the room ourselves, so onClose reports why

const maxBots = () => 2 * P.mode.team_size;
const pushBots = () => send({ type: "bots", count: bots.size, max: maxBots() });

function setupLobby() {
  addBot(RED);
  pushBots();
}

function teamCount(team) {
  return room.players.filter((p) => p.team && p.team.id === team).length;
}

// "Add bot" from the app: first free spot on red, then blue, else spectators
function requestAddBot() {
  if (!room || bots.size >= maxBots()) return pushBots();
  const ts = P.mode.team_size;
  addBot(teamCount(RED) < ts ? RED : teamCount(BLUE) < ts ? BLUE : SPEC);
  pushBots();
  pushPlayers();
}

// "Remove bot" from the app: a bot sitting in spectators first, else the newest bot
function requestRemoveBot() {
  if (!room || bots.size === 0) return pushBots();
  const ids = [...bots.keys()];
  const spec = ids.filter((id) => { const p = room.getPlayer(id); return !p || !p.team || p.team.id === SPEC; });
  const id = spec.length ? spec[spec.length - 1] : ids[ids.length - 1];
  bots.delete(id);
  try { room.fakePlayerLeave(id); } catch (e) { log(`bot leave failed: ${e.message}`); }
  pushBots();
  pushPlayers();
}

let exitProcess = () => setTimeout(() => process.exit(0), 300);

// Close the room if nobody joins within `seconds` (the app shows a countdown).
function startJoinTimer(seconds) {
  if (owner !== null || joinTimer) return;
  send({ type: "countdown", seconds });
  joinTimer = setTimeout(() => {
    joinTimer = null;
    closeReason = `nobody joined within ${Math.round(seconds)} s`;
    send({ type: "closed", reason: closeReason });
    try { room.leave(); } catch (e) { /* already closed */ }
    exitProcess();
  }, seconds * 1000);
}

function onJoin(p) {
  if (bots.has(p.id)) return;
  if (joinTimer) { clearTimeout(joinTimer); joinTimer = null; send({ type: "joined" }); }
  if (humans().length > cfg.maxPlayers) {
    after(() => room.kickPlayer(p.id, `Room is full (${cfg.maxPlayers} players)`, false));
    return;
  }
  const say = (msg) => after(() => room.sendAnnouncement(msg, p.id, 0x88ccff, "normal", 0));
  if (owner === null || !room.getPlayer(owner)) {
    owner = p.id;
    after(() => room.setPlayerAdmin(p.id, true));
    say("You're the room owner. Drag the bots and your friends between the teams (bots in spectators sit out), then press Start.");
  } else {
    say("Welcome! The room owner picks the teams and starts the game.");
  }
  pushPlayers();
}

function onLeave(p) {
  if (bots.has(p.id)) return;
  if (p.id === owner) {
    const next = humans().find((h) => h.id !== p.id);
    owner = next ? next.id : null;
    if (next) {
      after(() => room.setPlayerAdmin(next.id, true));
      after(() => room.sendAnnouncement("The owner left - you're the room owner now.", next.id, 0x88ccff, "normal", 0));
    }
  }
  pushPlayers();
}

function resetBots() { for (const b of bots.values()) { b.lastActions = [0, 0]; b.lastKey = -1; } }

function decide() {
  const gs = room.gameState;
  if (!gs) return;
  const ball = discInfo(gs.physicsState.discs[0]);
  const kickingTeam = gs.goalConcedingTeam && gs.goalConcedingTeam.id ? gs.goalConcedingTeam.id : RED;
  const active = room.players.filter((p) => p.disc && p.team && p.team.id !== SPEC);
  const m = P.mode;
  for (const [id, b] of bots) {
    const me = room.getPlayer(id);
    if (!me || !me.disc || !me.team || me.team.id === SPEC) continue;
    const team = me.team.id;
    const mates = active.filter((p) => p.id !== id && p.team.id === team).map((p) => discInfo(p.disc));
    const opps = active.filter((p) => p.team.id !== team).map((p) => discInfo(p.disc));
    if (opps.length === 0) continue;
    const g = {
      team, playState: gs.state, kickingTeam, timeS: gs.timeElapsed, timeLimitS: m.time_limit_min * 60,
      myScore: team === RED ? gs.redScore : gs.blueScore, theirScore: team === RED ? gs.blueScore : gs.redScore,
      lastActions: b.lastActions, goalX: m.goal_x, posScale: m.pos_scale, distScale: m.dist_scale,
    };
    const obs = pol.buildObs(discInfo(me.disc), mates, opps, ball, g);
    const lg = pol.logits(P, obs, mates.length, opps.length);
    const a = cfg.deterministic ? pol.argmax(lg) : pol.sample(lg);
    b.lastActions = [a, b.lastActions[0]];
    const [dx, dy, kick] = pol.actionKeys(a);
    const key = Utils.keyState((team === RED ? 1 : -1) * dx, dy, kick === 1); // policy acts in a mirrored frame
    if (key !== b.lastKey) {
      after(() => room.fakeSendPlayerInput(key, id));
      b.lastKey = key;
    }
  }
}

// ------------------------------------------------------------------ chat commands
function command(player, text) {
  const cmd = text.slice(1).toLowerCase().split(/\s+/)[0];
  const say = (msg, to = player.id) => after(() => room.sendAnnouncement(msg, to, 0x88ccff, "normal", 0));
  const isOwner = player.id === owner;
  if (cmd === "bot") say(`Bot: ${P.mode.title}, trained for ${P.mode.games_trained.toLocaleString()} games`);
  else if (cmd === "restart" && isOwner) after(() => { room.stopGame(); resetBots(); room.startGame(); });
  else say("Commands: !bot (what the bots are)" + (isOwner ? "  !restart (new game)" : ""));
}

function start(msg) {
  cfg = msg;
  P = msg.policy;
  const m = P.mode;
  log(`creating room for ${m.title}…`);
  Room.create(
    { name: msg.roomName || `haxballRL ${m.title}`, password: msg.password || undefined, showInRoomList: false,
      maxPlayerCount: Math.min(30, msg.maxPlayers + 2 * m.team_size), token: msg.token, noPlayer: true },  // humans + max bots
    {
      storage: { player_name: "haxballRL", avatar: "🤖" },
      onOpen: (r) => {
        room = r;
        if (m.stadium !== "Classic") {
          const st = Utils.getDefaultStadiums().find((s) => s.name === m.stadium);
          if (st) room.setCurrentStadium(st, 0); else log(`stadium ${m.stadium} not found, using Classic`);
        }
        after(() => {
          room.setTimeLimit(m.time_limit_min);
          room.setScoreLimit(0);
          if (typeof room.lockTeams === "function") room.lockTeams(); // only the host moves players between teams
        });
        setupLobby();
        room.onRoomLink = (link) => {
          send({ type: "link", link, password: msg.password || "" });
          startJoinTimer(JOIN_TIMEOUT_S);
        };
        room.onPlayerJoin = onJoin;
        room.onPlayerLeave = onLeave;
        room.onPlayerTeamChange = (id) => {
          const b = bots.get(id);
          if (b) { b.lastActions = [0, 0]; b.lastKey = -1; }
          pushPlayers();
        };
        room.onBeforeOperationReceived = (type, op) => {
          if (type === OperationType.SendChat && op.byId !== 0 && !bots.has(op.byId) && op.text && op.text.startsWith("!")) {
            command(room.getPlayer(op.byId), op.text);
            return false;
          }
          return true;
        };
        room.onGameStart = () => resetBots();
        room.onGameTick = () => { if (++tick % m.frame_skip === 0) decide(); };
        log("room is open");
        pushPlayers();
      },
      onClose: (reason) => {
        if (closeReason) return; // we closed it and already said why
        const code = reason && (reason.code !== undefined ? reason.code : reason);
        const text = code === 38 ? "HaxBall rejected the room token - press Play to solve a fresh captcha"
          : String((reason && (reason.message || reason.code)) || reason || "closed");
        send({ type: "closed", reason: text });
        room = null;
      },
    }
  );
}

function stop() {
  closeReason = "stopped";
  if (joinTimer) { clearTimeout(joinTimer); joinTimer = null; }
  try { if (room) room.leave(); } catch (e) { /* already closed */ }
  send({ type: "closed", reason: "stopped" });
  setTimeout(() => process.exit(0), 300);
}

if (process.parentPort) {
  process.parentPort.on("message", (e) => {
    const msg = e.data;
    if (msg.type === "start") {
      try { start(msg); } catch (err) { send({ type: "error", text: err.message }); }
    } else if (msg.type === "stop") stop();
    else if (msg.type === "add-bot") requestAddBot();
    else if (msg.type === "remove-bot") requestRemoveBot();
  });
}

// test hook (tests/host.test.js): drive the bot logic with a fake room
function _attachForTest(fakeRoom, policy, config, immediate, events) {
  room = fakeRoom; P = policy; cfg = config || {}; owner = null; closeReason = null;
  if (joinTimer) { clearTimeout(joinTimer); joinTimer = null; }
  exitProcess = () => {};
  if (events) send = (m) => events.push(m);
  if (immediate) after = (fn) => fn();
}

module.exports = { start, stop, decide, addBot, setupLobby, requestAddBot, requestRemoveBot, startJoinTimer, onJoin, onLeave, bots,
                   owner: () => owner, _attachForTest };
