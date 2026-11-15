// haxballRL desktop app: Electron main process.
// * the app window (renderer/index.html)
// * "Play": open HaxBall's headless-token page in a small window, the user solves the
//   captcha, we read the token off the page, then start the room host in its own process
//   (room/host.js, via utilityProcess) and forward its events to the window.
"use strict";
const { app, BrowserWindow, ipcMain, utilityProcess, shell, clipboard } = require("electron");
const fs = require("fs");
const path = require("path");

const TOKEN_URL = "https://www.haxball.com/headlesstoken";
let win = null;
let hostProc = null;

function policiesDir() { return path.join(__dirname, "policies"); }

function createWindow() {
  win = new BrowserWindow({
    width: 560, height: 540, resizable: false, title: "haxballRL", backgroundColor: "#ffffff",
    icon: path.join(__dirname, "assets", "icon.png"),
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false },
  });
  win.setMenuBarVisibility(false);
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
  // test hook: HAXBALLRL_SNAPSHOT=out.png saves a screenshot of the window and quits;
  // HAXBALLRL_SNAPSHOT_STEP = pick | selected | settings | live picks the screen
  if (process.env.HAXBALLRL_SNAPSHOT) {
    const step = process.env.HAXBALLRL_SNAPSHOT_STEP || "pick";
    const js = {
      pick: "",
      selected: "document.querySelector('.mode:nth-child(2)').click();",
      settings: "document.querySelector('.mode:nth-child(2)').click(); document.getElementById('play').click();",
      live: "document.querySelector('.mode:nth-child(1)').click(); document.getElementById('play').click(); show('live');",
      closed: "document.querySelector('.mode:nth-child(1)').click(); document.getElementById('play').click(); show('live');",
    }[step];
    win.webContents.once("did-finish-load", () => setTimeout(async () => {
      await win.webContents.executeJavaScript(js);
      if (step === "closed") toWindow({ type: "closed", reason: "nobody joined within 60 s" });
      if (step === "live") {
        toWindow({ type: "link", link: "https://www.haxball.com/play?c=AbCdEfGhIjK", password: "bot" });
        toWindow({ type: "bots", count: 1, max: 2 });
        toWindow({ type: "countdown", seconds: 60 });
      }
      setTimeout(async () => {
        const img = await win.webContents.capturePage();
        fs.writeFileSync(process.env.HAXBALLRL_SNAPSHOT, img.toPNG());
        app.quit();
      }, 600);
    }, 1200));
  }
}

function toWindow(msg) { if (win && !win.isDestroyed()) win.webContents.send("host-event", msg); }

// Open the token page and resolve with the token once the user has solved the captcha.
function getToken() {
  return new Promise((resolve, reject) => {
    const tw = new BrowserWindow({
      width: 420, height: 560, parent: win, modal: true, title: "Solve the captcha to create your room",
      webPreferences: { contextIsolation: true, nodeIntegration: false },
    });
    tw.setMenuBarVisibility(false);
    let done = false;
    const timer = setInterval(async () => {
      if (tw.isDestroyed()) return;
      try {
        const text = await tw.webContents.executeJavaScript("document.body ? document.body.innerText : ''");
        const m = text.match(/thr1\.[A-Za-z0-9._\-]+/);
        if (m) { done = true; clearInterval(timer); tw.close(); resolve(m[0]); }
      } catch (e) { /* page still loading */ }
    }, 500);
    tw.on("closed", () => { clearInterval(timer); if (!done) reject(new Error("captcha window closed")); });
    tw.loadURL(TOKEN_URL);
  });
}

function stopHost() {
  if (hostProc) {
    try { hostProc.postMessage({ type: "stop" }); } catch (e) { /* gone */ }
    const p = hostProc;
    setTimeout(() => { try { p.kill(); } catch (e) { /* gone */ } }, 1500);
    hostProc = null;
  }
}

ipcMain.handle("modes", () => JSON.parse(fs.readFileSync(path.join(policiesDir(), "index.json"), "utf8")));

ipcMain.handle("start-room", async (_e, opts) => {
  stopHost();
  toWindow({ type: "status", text: "Solve the captcha in the popup to get a room token…" });
  let token;
  try { token = await getToken(); } catch (e) { toWindow({ type: "error", text: "Cancelled: the captcha window was closed." }); return; }
  toWindow({ type: "status", text: "Creating your room…" });
  const policy = JSON.parse(fs.readFileSync(path.join(policiesDir(), `${opts.mode}.json`), "utf8"));
  hostProc = utilityProcess.fork(path.join(__dirname, "room", "host.js"), [], { serviceName: "haxballRL room" });
  hostProc.on("message", (msg) => toWindow(msg));
  hostProc.on("exit", (code) => { toWindow({ type: "closed", reason: code ? `room process exited (${code})` : "stopped" }); });
  hostProc.postMessage({ type: "start", token, policy, password: opts.password, maxPlayers: opts.maxPlayers,
                         roomName: opts.roomName });
});

ipcMain.handle("stop-room", () => { stopHost(); });
ipcMain.handle("add-bot", () => { if (hostProc) hostProc.postMessage({ type: "add-bot" }); });
ipcMain.handle("remove-bot", () => { if (hostProc) hostProc.postMessage({ type: "remove-bot" }); });
ipcMain.handle("open-link", (_e, url) => { if (/^https:\/\/www\.haxball\.com\//.test(url)) shell.openExternal(url); });
ipcMain.handle("copy", (_e, text) => clipboard.writeText(String(text)));

// test hook: HAXBALLRL_HOSTTEST=1 starts the room process with an invalid token, prints its
// events and quits (checks node-haxball + its native WebRTC module load inside Electron)
if (process.env.HAXBALLRL_HOSTTEST) {
  app.whenReady().then(() => {
    const p = utilityProcess.fork(path.join(__dirname, "room", "host.js"));
    p.on("message", (m) => console.log("host event:", JSON.stringify(m)));
    p.on("exit", (c) => console.log("host exit", c));
    const policy = JSON.parse(fs.readFileSync(path.join(policiesDir(), "3v3-big.json"), "utf8"));
    p.postMessage({ type: "start", token: "thr1.invalid", policy, password: "bot", maxPlayers: 4 });
    p.postMessage({ type: "add-bot" }); p.postMessage({ type: "add-bot" }); p.postMessage({ type: "remove-bot" });
    setTimeout(() => { p.kill(); app.quit(); }, 12000);
  });
} else app.whenReady().then(() => {
  if (process.platform === "darwin" && app.dock) app.dock.setIcon(path.join(__dirname, "assets", "icon.png"));
  createWindow();
  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on("window-all-closed", () => { stopHost(); app.quit(); });
app.on("before-quit", stopHost);
