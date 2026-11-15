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
  // bottom bar: the one or two buttons for this step
  $("play").hidden = step !== "pick";
  $("back").hidden = $("create").hidden = step !== "settings";
  $("closeRoom").hidden = step !== "live";
}
// the room ended on its own (timeout, rejected token, captcha closed): go back to the bot list
// and say why in a small note there
function endRoom(text) {
  stopCountdown();
  $("botBox").hidden = true;
  $("notice").textContent = text;
  $("notice").hidden = false;
  show("pick");
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

let countdownTimer = null;
function stopCountdown() { clearInterval(countdownTimer); countdownTimer = null; $("countdown").hidden = true; }
function startCountdown(seconds) {
  stopCountdown();
  const end = Date.now() + seconds * 1000;
  const tick = () => {
    const left = Math.max(0, Math.ceil((end - Date.now()) / 1000));
    $("countdown").textContent = `Closes in ${left}s if nobody joins`;
    if (left === 0) stopCountdown();
  };
  $("countdown").hidden = false;
  tick();
  countdownTimer = setInterval(tick, 250);
}

api.onEvent((msg) => {
  if (msg.type === "status") setStatus(msg.text);
  else if (msg.type === "error") endRoom(msg.text);
  else if (msg.type === "link") {
    link = msg.link;
    $("link").textContent = msg.link;
    $("pw").textContent = msg.password || "(no password)";
    $("linkBox").hidden = false;
    setStatus("Your room is ready. Join with the link first and you'll be the room owner.");
    $("botBox").hidden = false;
  } else if (msg.type === "countdown") startCountdown(msg.seconds);
  else if (msg.type === "joined") stopCountdown();
  else if (msg.type === "bots") {
    $("botCount").textContent = `${msg.count} / ${msg.max}`;
    $("addBot").disabled = msg.count >= msg.max;
    $("removeBot").disabled = msg.count <= 0;
  } else if (msg.type === "closed") {
    if (msg.reason !== "stopped" && !$("live").hidden) endRoom(`Room closed: ${msg.reason}.`);
  }
});

$("play").onclick = () => {
  if (!selected) return;
  $("notice").hidden = true;
  $("settingsTitle").textContent = `${selected.title} room`;
  $("maxPlayers").value = DEFAULT_MAX[selected.team_size] || 4;
  show("settings");
};
$("back").onclick = () => show("pick");
$("create").onclick = async () => {
  link = "";
  $("linkBox").hidden = true;
  $("botBox").hidden = true;
  stopCountdown();
  $("liveTitle").textContent = `${selected.title} room`;
  setStatus("Starting…");
  show("live");
  await api.startRoom({
    mode: selected.key, password: $("password").value.trim(), roomName: $("roomName").value.trim(),
    maxPlayers: Math.max(1, Math.min(12, parseInt($("maxPlayers").value, 10) || 1)),
  });
};
$("closeRoom").onclick = async () => { stopCountdown(); await api.stopRoom(); show("pick"); };
$("open").onclick = () => link && api.openLink(link);
$("addBot").onclick = () => api.addBot();
$("removeBot").onclick = () => api.removeBot();
$("copy").onclick = async () => {
  if (!link) return;
  await api.copy(link);
  $("copy").textContent = "Copied!";
  setTimeout(() => ($("copy").textContent = "Copy link"), 1200);
};

api.modes().then((m) => { modes = m; renderModes(); show("pick"); });
