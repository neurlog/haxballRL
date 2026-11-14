// Host a private HaxBall room where a trained haxballrl policy plays 1v1 against you.
//
//   1. python headless-bot/export_policy.py 1v1-classic/runs/main/league/champion.pt
//   2. put HEADLESS_TOKEN=<token from https://www.haxball.com/headlesstoken> in headless-bot/.env
//   3. node headless-bot/bot.js        -> prints a room link; open it in your browser
//
// Optional env: ROOM_NAME, ROOM_PASS (default "bot"), BOT_PING_MS (default 40: delay the bot's
// inputs like the ping it trained with), DETERMINISTIC=1 (argmax instead of sampling).
//
// Room structure adapted from HaxballGym's headless-bot (MIT). node-haxball loads the official
// HaxBall engine; the policy and observation live in policy.js (parity-tested vs Python).
"use strict";
const fs = require("fs");
const path = require("path");
const { Room, Utils, OperationType } = require("node-haxball")();
const pol = require("./policy");

// --- .env (gitignored): HEADLESS_TOKEN=... ; real environment variables win
try {
  for (const line of fs.readFileSync(path.join(__dirname, ".env"), "utf8").split("\n")) {
    const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!m || process.env[m[1]] !== undefined) continue;
    let v = m[2];
    if (v.length >= 2 && (v[0] === '"' || v[0] === "'") && v[v.length - 1] === v[0]) v = v.slice(1, -1);
    process.env[m[1]] = v;
  }
} catch { /* no .env */ }

const TOKEN = process.env.HEADLESS_TOKEN;
if (!TOKEN) {
  console.error("Set HEADLESS_TOKEN (get one at https://www.haxball.com/headlesstoken), e.g. in headless-bot/.env");
  process.exit(1);
}
const CFG = {
  roomName: process.env.ROOM_NAME || "haxballrl bot (private)",
  password: process.env.ROOM_PASS || "bot",
  frameSkip: 2, // the policy decides every 2 ticks, as in training
  pingTicks: parseFloat(process.env.BOT_PING_MS || "40") * 60 / 1000,
  deterministic: process.env.DETERMINISTIC === "1",
  timeLimitMin: 3,
  goalX: 370,
};
const BOT_ID = 65535;
const { RED, BLUE } = pol;
const SPEC = 0;

const POLICY_PATH = path.join(__dirname, "policy.json");
let P = JSON.parse(fs.readFileSync(POLICY_PATH, "utf8"));
const reload = () => { P = JSON.parse(fs.readFileSync(POLICY_PATH, "utf8")); };

// --- bot control
let tick = 0, lastActions = [0, 0], pending = [], lastKey = -1;
const after = (fn) => Utils.runAfterGameTick(fn);
const humans = (room) => room.players.filter((p) => p.id !== BOT_ID);
const announce = (room, msg, to = null, color = 0x88ccff) =>
  after(() => room.sendAnnouncement(msg, to, color, "normal", 0));
const discInfo = (d) => ({ x: d.pos.x, y: d.pos.y, vx: d.speed.x, vy: d.speed.y });

function resetBotState() { lastActions = [0, 0]; pending = []; lastKey = -1; }

function decide(room) {
  const gs = room.gameState;
  const me = room.getPlayer(BOT_ID);
  if (!gs || !me || !me.disc || !me.team || me.team.id === SPEC) return;
  const team = me.team.id;
  const others = room.players.filter((p) => p.id !== BOT_ID && p.disc && p.team && p.team.id !== SPEC);
  const mates = others.filter((p) => p.team.id === team).map((p) => discInfo(p.disc));
  const opps = others.filter((p) => p.team.id !== team).map((p) => discInfo(p.disc));
  if (opps.length === 0) return;
  const ball = discInfo(gs.physicsState.discs[0]);
  const kickingTeam = gs.goalConcedingTeam && gs.goalConcedingTeam.id ? gs.goalConcedingTeam.id : RED;
  const g = {
    team, playState: gs.state, kickingTeam, timeS: gs.timeElapsed, timeLimitS: CFG.timeLimitMin * 60,
    myScore: team === RED ? gs.redScore : gs.blueScore, theirScore: team === RED ? gs.blueScore : gs.redScore,
    lastActions, goalX: CFG.goalX,
  };
  const obs = pol.buildObs(discInfo(me.disc), mates, opps, ball, g);
  const lg = pol.logits(P, obs, mates.length, opps.length);
  const a = CFG.deterministic ? pol.argmax(lg) : pol.sample(lg);
  lastActions = [a, lastActions[0]];
  const [dx, dy, kick] = pol.actionKeys(a);
  const sx = team === RED ? 1 : -1; // the policy acts in a mirrored frame: always attacking +x
  const key = Utils.keyState(sx * dx, dy, kick === 1);
  // emulate the training ping: inputs reach the engine ~pingTicks after the observation
  // (a fake player's input already lands one tick later, so subtract that tick)
  const d = Math.max(0, CFG.pingTicks - 1);
  const delay = Math.floor(d) + (Math.random() < d - Math.floor(d) ? 1 : 0);
  pending.push({ at: tick + delay, key });
}

function flush(room) {
  while (pending.length && pending[0].at <= tick) {
    const { key } = pending.shift();
    if (key !== lastKey) {
      room.fakeSendPlayerInput(key, BOT_ID);
      lastKey = key;
    }
  }
}

function onTick(room) {
  tick++;
  if (tick % CFG.frameSkip === 0) decide(room);
  after(() => flush(room));
}

function startIfReady(room) {
  const opp = humans(room).find((p) => p.team && p.team.id !== SPEC);
  if (opp && !room.gameState) after(() => { resetBotState(); room.startGame(); });
}

function command(room, player, text) {
  const [cmd] = text.slice(1).toLowerCase().split(/\s+/);
  switch (cmd) {
    case "restart": after(() => { room.stopGame(); resetBotState(); room.startGame(); }); break;
    case "start": after(() => { resetBotState(); room.startGame(); }); break;
    case "stop": after(() => room.stopGame()); break;
    case "swap":
      after(() => {
        const botTeam = room.getPlayer(BOT_ID).team.id;
        room.setPlayerTeam(BOT_ID, botTeam === RED ? BLUE : RED);
        for (const h of humans(room)) if (h.team && h.team.id !== SPEC) room.setPlayerTeam(h.id, botTeam);
      });
      break;
    case "reload":
      try { reload(); announce(room, `Reloaded policy: ${P.name}`); } catch (e) { announce(room, `Reload failed: ${e.message}`, player.id, 0xff8888); }
      break;
    case "who": announce(room, `Bot policy: ${P.name}`, player.id); break;
    default: announce(room, "Commands: !start !stop !restart !swap !reload !who", player.id);
  }
}

Room.create(
  { name: CFG.roomName, password: CFG.password, showInRoomList: false, maxPlayerCount: 8, token: TOKEN, noPlayer: true },
  {
    storage: { player_name: "haxballrl-host" },
    onOpen: (room) => {
      room.fakePlayerJoin(BOT_ID, "RL bot", "tr", "", "fake-conn", "fake-auth");
      after(() => {
        room.setPlayerTeam(BOT_ID, RED);
        room.setTimeLimit(CFG.timeLimitMin);
        room.setScoreLimit(0);
      });
      room.onRoomLink = (link) => console.log(`\n  >>> JOIN: ${link}\n  >>> password: ${CFG.password}\n  >>> policy: ${P.name}\n`);
      room.onPlayerJoin = (player) => {
        console.log(`+ ${player.name} (id ${player.id})`);
        after(() => {
          room.setPlayerAdmin(player.id, true);
          const taken = humans(room).some((p) => p.id !== player.id && p.team && p.team.id === BLUE);
          room.setPlayerTeam(player.id, taken ? SPEC : BLUE);
        });
        announce(room, "You're admin and on blue. The bot is red. Type !help for commands.", player.id);
        startIfReady(room);
      };
      room.onPlayerLeave = () => {
        if (!humans(room).some((p) => p.team && p.team.id !== SPEC) && room.gameState) after(() => room.stopGame());
      };
      room.onBeforeOperationReceived = (type, msg) => {
        if (type === OperationType.SendChat && msg.byId !== 0 && msg.text && msg.text.startsWith("!")) {
          command(room, room.getPlayer(msg.byId), msg.text);
          return false;
        }
        return true;
      };
      room.onGameStart = () => resetBotState();
      room.onGameEnd = () => setTimeout(() => startIfReady(room), 3000);
      room.onGameTick = () => onTick(room);
    },
    onClose: (msg) => {
      console.error("Room closed:", (msg && (msg.code || msg)) || "(unknown)");
      process.exit(1);
    },
  }
);
