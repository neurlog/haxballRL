# headless-bot: play your trained agent in real HaxBall

[node-haxball](https://github.com/wxyz-abcd/node-haxball) loads the **official** HaxBall
engine in Node.js. `bot.js` uses it to host a private room. Your policy plays in that room as
a bot player, and you join from the normal browser client. HaxballGym's physics is a
bit-exact port of that same engine, so the bot plays under the physics it trained on.

| file | what it does |
|---|---|
| `export_policy.py` | turns a checkpoint into `policy.json` (plain weights, no torch needed) |
| `policy.js` | observation builder and network forward pass in JavaScript (mirrors `hax/env.py` and `hax/model.py`) |
| `parity_test.py` | checks that `policy.js` gives the same observations and logits as Python |
| `bot.js` | hosts the room, reads the live game state every 2 ticks, runs the policy, sends inputs |

## One-time setup

1. **Node.js 18 or newer.** This machine already has Node 26: run `node --version` to check.
   Otherwise, `brew install node`.
2. **Install node-haxball.** It is already installed in `headless-bot/node_modules`. On a new machine:
   ```bash
   cd headless-bot && npm install
   ```
3. **Get a headless token.** Open https://www.haxball.com/headlesstoken in your browser and
   solve the reCAPTCHA. It shows a token like `thr1.AAAAA...`. Tokens are single-use and expire
   after a few minutes, so get a fresh one each time you start the bot.
4. Put the token in `headless-bot/.env` (this file is gitignored):
   ```
   HEADLESS_TOKEN=thr1.AAAAA...
   ```

## Every time

```bash
source .venv/bin/activate
python headless-bot/export_policy.py 1v1-classic/runs/main/league/champion.pt
python headless-bot/parity_test.py 1v1-classic/runs/main/league/champion.pt   # optional; should say PASS
node headless-bot/bot.js
```

The bot prints `JOIN: https://www.haxball.com/play?c=...` and a password (default `bot`).
Open that link and you join as admin on blue. The bot is red, and the game starts on its own:
3 minutes, Classic map.

In-room chat commands: `!start` `!stop` `!restart` `!swap` (switch sides) `!reload` (re-read
`policy.json` after a new export) `!who`.

Options, set as environment variables or in `.env`:

| variable | default | meaning |
|---|---|---|
| `ROOM_NAME` | `haxballrl bot (private)` | room title |
| `ROOM_PASS` | `bot` | room password |
| `BOT_PING_MS` | `0` | Extra delay on the bot's inputs. Leave it at 0 for models trained with the default `--ping-ms 0`; set it to the training ping if you trained with one. |
| `DETERMINISTIC` | off | `1` = always pick the most likely action. By default actions are sampled, as in training. |

## How it lines up with training

* **Decisions:** every 2 ticks (30 per second), the same as `--frame-skip 2`.
* **Observation:** built from `room.gameState`.
  * Disc positions and speeds.
  * `state`: 0 = kickoff, 1 = playing, 2 = goal.
  * `goalConcedingTeam`: tells who kicks off.
  * `timeElapsed` and the score.
  * The bot's own last two actions.
* **Clock:** the real engine stops the clock during kickoffs and goal celebrations. I checked
  this in node-haxball's offline sandbox, and `hax/env.py` does the same.
* **Mirroring:** the policy always "attacks +x", so for a blue bot the x direction is flipped
  on the way in and on the way out.

## Etiquette

Keep rooms private (the default, `showInRoomList: false`) and use them to play against your own
bot. Putting a bot into public rooms against people who don't know it's a bot breaks the
community's rules and gets banned.
