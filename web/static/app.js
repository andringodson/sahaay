/* Sahaay front end.
 *
 * No framework on purpose. The whole UI is a WebSocket feed appending to two
 * lists, and shipping a build step would mean the one-click installer needs
 * Node on a Windows ARM64 machine. This file stays readable instead.
 */

const $ = (id) => document.getElementById(id);

const state = {
  ws: null,
  running: false,
  captions: new Map(),   // index -> {el, srcEl, trEl}
  glossCount: 0,
  capScale: Number(localStorage.getItem("sahaay.capScale") || 1),
  reconnectDelay: 500,
};

/* ---------- caption sizing (persisted per viewer) ---------- */

function applyScale() {
  document.documentElement.style.setProperty("--cap-scale", state.capScale.toFixed(2));
  try { localStorage.setItem("sahaay.capScale", String(state.capScale)); } catch (_) {}
}
applyScale();

$("bigger").onclick = () => { state.capScale = Math.min(2.2, state.capScale + 0.15); applyScale(); };
$("smaller").onclick = () => { state.capScale = Math.max(0.7, state.capScale - 0.15); applyScale(); };

/* ---------- toast ---------- */

let toastTimer = null;
function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 7000 : 3500);
}

/* ---------- device badge ---------- */

function renderDevice(device) {
  if (!device) return;
  const badge = $("device");
  $("device-label").textContent = device.provider_label || device.provider || "unknown";
  badge.classList.remove("badge-npu", "badge-cpu", "badge-muted");
  // The badge is the honest answer to "is this really using the NPU?".
  badge.classList.add(device.npu_active ? "badge-npu" : "badge-cpu");
  badge.title = device.fallback_reason
    ? `Fallback: ${device.fallback_reason}`
    : `Running on ${device.provider}`;

  const rows = [
    ["Provider", device.provider_label],
    ["Runtime", `onnxruntime ${device.ort_version}`],
    ["Arch", device.machine + (device.is_arm64 ? " (ARM64)" : "")],
    ["Available", (device.available_providers || []).length
      ? device.available_providers.map((p) => p.replace("ExecutionProvider", "")).join(", ")
      : "—"],
  ];
  $("devinfo").innerHTML = rows
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v ?? "—"))}</dd>`)
    .join("");
}

/* ---------- captions ---------- */

function ensureCaption(index) {
  if (state.captions.has(index)) return state.captions.get(index);

  $("empty").style.display = "none";
  const li = document.createElement("li");
  li.className = "cap";

  const src = document.createElement("p");
  src.className = "cap-src";

  const tr = document.createElement("p");
  tr.className = "cap-tr";
  tr.hidden = true;

  const foot = document.createElement("div");
  foot.className = "cap-foot";

  li.append(src, tr, foot);
  $("captions").appendChild(li);

  const entry = { el: li, srcEl: src, trEl: tr, footEl: foot };
  state.captions.set(index, entry);
  return entry;
}

function atBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
}

function onCaption(msg) {
  const list = $("captions");
  const stick = atBottom(list);

  // A partial shares the index of the caption that will replace it, so the
  // line doesn't jump position when the final text lands.
  const entry = ensureCaption(msg.index);
  entry.srcEl.textContent = msg.text;
  entry.srcEl.classList.remove("cap-partial");

  const bits = [];
  if (msg.start_s != null) bits.push(fmtTime(msg.start_s));
  if (msg.latency_ms != null) bits.push(`${Math.round(msg.latency_ms)} ms`);
  if (msg.rtf != null) bits.push(`RTF ${Number(msg.rtf).toFixed(2)}`);
  if (msg.language) bits.push(msg.language);
  entry.footEl.textContent = bits.join(" · ");

  $("count").textContent = `${state.captions.size} lines`;
  if (stick) list.scrollTop = list.scrollHeight;
}

function onPartial(msg) {
  const entry = ensureCaption(msg.index);
  if (!entry.srcEl.textContent) {
    entry.srcEl.textContent = msg.text;
    entry.srcEl.classList.add("cap-partial");
  }
}

function onTranslation(msg) {
  const entry = state.captions.get(msg.index);
  if (!entry) return;
  if (msg.passthrough) {
    // Say so rather than showing English text under a Hindi heading.
    entry.trEl.textContent = "translation model not installed";
    entry.trEl.style.opacity = "0.5";
  } else {
    entry.trEl.textContent = msg.text;
    entry.trEl.style.opacity = "";
  }
  entry.trEl.hidden = false;
}

/* ---------- glossary ---------- */

function onGloss(msg) {
  const li = document.createElement("li");
  li.className = "gloss";
  li.innerHTML =
    `<div class="gloss-term">${esc(msg.term)}</div>` +
    `<p class="gloss-def">${esc(msg.explanation)}</p>` +
    `<div class="gloss-meta">${esc(msg.backend || "")}` +
    (msg.latency_ms ? ` · ${Math.round(msg.latency_ms)} ms` : "") +
    `</div>`;
  $("glossary").prepend(li);
  state.glossCount += 1;
  $("gloss-count").textContent = String(state.glossCount);
}

/* ---------- metrics ---------- */

function onLevel(msg) {
  const fill = $("level-fill");
  if (!fill) return;
  // Square root, not linear: speech sits between 0.01 and 0.1 rms, and a
  // linear bar would sit near zero for all of it and tell you nothing.
  const rms = Math.max(0, Number(msg.rms) || 0);
  const pct = Math.min(100, Math.round(Math.sqrt(rms / 0.12) * 100));
  fill.style.width = pct + "%";

  const quiet = rms < 0.0002;
  $("level").classList.toggle("silent", state.running && quiet);
  $("level").title = quiet
    ? "No audio reaching the app — check the output device is not muted"
    : `Input level ${rms.toFixed(4)} rms`;
}

function onMetric(msg) {
  if (msg.realtime_factor == null) return;
  const rtf = Number(msg.realtime_factor);
  const el = $("rtf");
  el.textContent = `RTF ${rtf.toFixed(2)}`;
  el.classList.toggle("good", rtf > 0 && rtf < 1);
  el.classList.toggle("bad", rtf >= 1);
  el.title = rtf < 1
    ? "Transcribing faster than real time — captions keep up"
    : "Slower than real time — captions will drift behind";
}

/* ---------- notes ---------- */

let lastMarkdown = "";

function onNotes(msg) {
  lastMarkdown = msg.markdown || "";
  $("notes-title").textContent = msg.title || "Session notes";
  $("notes-path").textContent = msg.path ? `Saved to ${msg.path}` : "";

  const parts = [];
  if (msg.summary) parts.push(renderMarkdown(msg.summary));

  if (msg.glossary && msg.glossary.length) {
    parts.push("<h2>Glossary</h2><ul>" +
      msg.glossary.map((g) => `<li><strong>${esc(g.term)}</strong> — ${esc(g.explanation)}</li>`).join("") +
      "</ul>");
  }
  if (msg.quiz && msg.quiz.length) {
    parts.push("<h2>Self-test</h2>" + msg.quiz.map((q, i) =>
      `<div class="q">${i + 1}. ${esc(q.q)}` +
      `<details><summary>Answer</summary>${esc(q.a)}</details></div>`).join(""));
  }

  $("notes-body").innerHTML = parts.join("") || "<p>No notes were generated.</p>";
  $("notes").hidden = false;
}

$("notes-close").onclick = () => { $("notes").hidden = true; };
$("notes-copy").onclick = (e) => {
  e.preventDefault();
  navigator.clipboard.writeText(lastMarkdown)
    .then(() => toast("Markdown copied"))
    .catch(() => toast("Could not copy", true));
};

/* A deliberately small Markdown subset: headings, bullets, bold. The notes
   come from our own prompt, so supporting the full spec would be dead code. */
function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "";
  let inList = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("## ")) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h2>${esc(line.slice(3))}</h2>`;
    } else if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${bold(esc(line.slice(2)))}</li>`;
    } else if (line) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<p>${bold(esc(line))}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

const bold = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

/* ---------- helpers ---------- */

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/* ---------- control ---------- */

const IDLE_HINT = $("empty") ? $("empty").innerHTML : "";

function setRunning(running) {
  state.running = running;
  const btn = $("toggle");
  btn.textContent = running ? "Stop" : "Start";
  btn.classList.toggle("recording", running);
  btn.disabled = false;

  // A caption only lands once the speaker pauses, so the first one takes a
  // few seconds however fast the models are. Saying so is the difference
  // between "it is working" and "it is broken".
  const empty = $("empty");
  if (empty && !state.captions.size) {
    empty.innerHTML = running
      ? "Listening… the first caption appears when the speaker pauses. " +
        "Watch the level meter in the header: if it is moving, sound is reaching the app."
      : IDLE_HINT;
  }
}

$("toggle").onclick = async () => {
  const btn = $("toggle");
  btn.disabled = true;
  const endpoint = state.running ? "/api/stop" : "/api/start";
  btn.textContent = state.running ? "Saving…" : "Starting…";
  try {
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "request failed");
    setRunning(Boolean(data.running));
    if (endpoint === "/api/stop") {
      toast(data.saved ? "Session saved" : "Nothing captured — no notes written");
    }
  } catch (err) {
    toast(String(err.message || err), true);
    setRunning(state.running);
  }
};

$("language").onchange = async (e) => {
  const code = e.target.value;
  await fetch(`/api/language/${code}`, { method: "POST" });
  toast(`Captions will be translated into ${e.target.selectedOptions[0].textContent}`);
};

/* ---------- websocket ---------- */

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => { state.reconnectDelay = 500; };

  ws.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch (_) { return; }
    switch (msg.kind) {
      case "caption": onCaption(msg); break;
      case "partial": onPartial(msg); break;
      case "translation": onTranslation(msg); break;
      case "gloss": onGloss(msg); break;
      case "metric": onMetric(msg); break;
      case "level": onLevel(msg); break;
      case "notes": onNotes(msg); break;
      case "status": onStatus(msg); break;
      case "error": toast(msg.message, true); break;
    }
  };

  ws.onclose = () => {
    // The server restarting during development should not mean a manual
    // refresh; back off so a real outage doesn't spin.
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(8000, state.reconnectDelay * 2);
  };
}

function onStatus(msg) {
  renderDevice(msg.device);
  if (typeof msg.running === "boolean") setRunning(msg.running);
  if (msg.metrics) onMetric(msg.metrics);
  if (msg.stage === "loading") {
    $("toggle").disabled = true;
    $("toggle").textContent = "Loading…";
    toast(`Loading ${msg.detail || "models"}…`);
  } else if (msg.stage === "warming") {
    $("toggle").disabled = true;
    $("toggle").textContent = "Warming up…";
    toast("Warming the models up so your first caption is not the slow one…");
  } else if (msg.stage === "ready") {
    $("toggle").disabled = false;
    $("toggle").textContent = state.running ? "Stop" : "Start";
    toast("Ready — everything runs on this device");
  } else if (msg.stage === "summarising") {
    toast("Writing your notes…");
  }
  if (msg.mock) {
    $("device").title = "Mock mode: no models are being executed";
  }
}

/* ---------- boot ---------- */

async function boot() {
  try {
    const status = await (await fetch("/api/status")).json();
    renderDevice(status.device);
    setRunning(status.running);

    const sel = $("language");
    sel.innerHTML = (status.languages || [])
      .map((l) => `<option value="${esc(l.code)}">${esc(l.name)}</option>`)
      .join("");
    sel.value = status.target_language || "hi";
  } catch (err) {
    toast("Could not reach the local server", true);
  }
  connect();
}

/* Space toggles start/stop unless the user is in a form control. */
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(document.activeElement.tagName)) {
    e.preventDefault();
    $("toggle").click();
  }
  if (e.key === "Escape" && !$("notes").hidden) $("notes").hidden = true;
});

boot();
