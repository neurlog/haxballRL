# haxballRL desktop app

A small macOS app to play the trained bots in real HaxBall without a terminal:
pick a mode, press **Play**, solve one captcha, and get a room link.

## Use

1. Open `dist/haxballRL-0.1.0-arm64.dmg` and drag **haxballRL** to Applications.
2. Under **Bots**, click **1v1 Classic** or **3v3 Big** (click it again to unselect), then **Play**.
3. Set a password, max players and room name, then press **Create room**.
4. A small window shows HaxBall's headless-token page. Solve the captcha and the app reads
   the token automatically. Every room needs a fresh one; that's HaxBall's rule.
5. The app shows the room link and password. **Join first** and you become the room owner:
   * The room opens as a lobby with bots on both teams (1v1: one each, 3v3: three each).
     Nothing starts by itself.
   * Everyone joins as a spectator. As owner, drag players in HaxBall's lobby: move a bot to
     spectators and drag yourself or a friend into its place. Bots on a team play; bots in
     spectators sit out. Then press **Start**.
   * If the owner leaves, the next person in the room becomes owner.
   * Chat commands: `!bot` (what the bots are), `!restart` (owner only).
6. **Close room** ends it.

Rooms are always private (not in the public room list).

The app isn't signed with an Apple developer certificate. On another Mac, the first launch
needs **right-click -> Open** (or System Settings -> Privacy & Security -> Open Anyway).

## Develop

```bash
cd app
npm install
npm start                  # run the app
node tests/host.test.js    # bot seating / decision test with a fake room
npm run dist               # build dist/haxballRL-<version>-arm64.dmg
```

`HAXBALLRL_HOSTTEST=1 npm start` starts the room process with a fake token and prints its
events. This checks that node-haxball and its native WebRTC module load inside Electron.

## Files

| path | what |
|---|---|
| `main.js` | Electron main process: window, captcha/token window, starts the room process |
| `preload.js` | the small API the window may call |
| `renderer/` | the window: header replay (`pitch.js`), mode cards and room panel (`renderer.js`) |
| `room/host.js` | hosts the room with node-haxball and runs the bots (one process per room) |
| `room/policy.js` | observation + network in JS; must stay identical to `headless-bot/policy.js` (parity-tested) |
| `policies/` | bot weights + mode settings, built from the trained models |
| `assets/` | logo and the recorded 3v3 match behind the header |
| `tools/build_assets.py` | rebuilds `policies/` and `assets/` from the models |

## Adding a game mode

Train it, put the champion in `<mode>/models/`, add an entry to `MODES` in
`tools/build_assets.py`, run `python app/tools/build_assets.py policies` from the repo root,
then `npm run dist`. The new card appears automatically. Any team size works, and stadiums
must be one of HaxBall's default maps.
