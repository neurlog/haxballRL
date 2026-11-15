"use strict";
const $ = (id) => document.getElementById(id);
const api = window.haxballrl;
let modes = [], selected = null, link = "";
const DEFAULT_MAX = { 1: 4, 3: 6 };  // humans allowed in the room (extra people spectate)

// step: "pick" -> "settings" -> "live"
function show(step) {
  $("pick").hidden = step !== "pick";
  $("settings").hidden = step !== "settings";
  $("live").hidden = step !== "live";
  $("play").hidden = step !== "pick";
  $("create").hidden = step !== "settings";
  $("back").hidden = step !== "settings";
  $("stop").hidden = step !== "live";
}

function renderModes() {
  $("modes").innerHTML = "";
  for (const m of modes) {
    const el = document.createElement("div");
    el.className = "mode" + (selected && selected.key === m.key ? " selected" : "");
    el.textContent = m.title;
    el.onclick = () => {  // click to select, click the selected one again to unselect
      selected = selected && selected.key === m.key ? null : m;
      $("play").disabled = !selected;
      renderModes();
    };
    $("modes").appendChild(el);
  }
}

function setStatus(text, error = false) { $("status").textContent = text; $("status").className = "status" + (error ? " error" : ""); }

api.onEvent((msg) => {
  if (msg.type === "status") setStatus(msg.text);
  else if (msg.type === "error") setStatus(msg.text, true);
  else if (msg.type === "link") {
    link = msg.link;
    $("link").textContent = msg.link;
    $("pw").textContent = msg.password || "(no password)";
    $("linkBox").hidden = false;
    setStatus("Your room is ready. Join with the link first and you'll be the room owner: drag the bots and your friends between the teams, then press Start.");
  } else if (msg.type === "closed") {
    setStatus(`Room closed (${msg.reason}).`, msg.reason !== "stopped");
    $("stop").textContent = "Back";
  }
});

$("play").onclick = () => {
  if (!selected) return;
  $("settingsTitle").textContent = `${selected.title} room`;
  $("maxPlayers").value = DEFAULT_MAX[selected.team_size] || 4;
  show("settings");
};
$("back").onclick = () => show("pick");
$("create").onclick = async () => {
  link = "";
  $("linkBox").hidden = true;
  $("liveTitle").textContent = `${selected.title} room`;
  $("stop").textContent = "Close room";
  setStatus("Starting…");
  show("live");
  await api.startRoom({
    mode: selected.key, password: $("password").value.trim(), roomName: $("roomName").value.trim(),
    maxPlayers: Math.max(1, Math.min(12, parseInt($("maxPlayers").value, 10) || 1)),
  });
};
$("stop").onclick = async () => { await api.stopRoom(); show("pick"); };
$("open").onclick = () => link && api.openLink(link);
$("copy").onclick = async () => {
  if (!link) return;
  await api.copy(link);
  $("copy").textContent = "Copied!";
  setTimeout(() => ($("copy").textContent = "Copy link"), 1200);
};

api.modes().then((m) => { modes = m; renderModes(); show("pick"); });
