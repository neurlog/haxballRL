// Exposes a small, safe API to the app window (no Node access in the page itself).
"use strict";
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("haxballrl", {
  modes: () => ipcRenderer.invoke("modes"),
  replay: () => ipcRenderer.invoke("replay"),
  startRoom: (opts) => ipcRenderer.invoke("start-room", opts),
  stopRoom: () => ipcRenderer.invoke("stop-room"),
  openLink: (url) => ipcRenderer.invoke("open-link", url),
  copy: (text) => ipcRenderer.invoke("copy", text),
  onEvent: (fn) => ipcRenderer.on("host-event", (_e, msg) => fn(msg)),
});
