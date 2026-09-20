/* Replay harness for the hosted demo.
 *
 * app.js is the product's UI, copied byte for byte by scripts/build_web.py.
 * It expects a local server: GET /api/status, POST /api/start, and a
 * WebSocket at /ws. None of those exist on a static host, and they should
 * not - the whole claim is that inference happens on the user's machine.
 *
 * So this replaces the transport and nothing else. It stubs fetch and
 * WebSocket, then feeds app.js a recording made by scripts/record_demo.py:
 * the actual events the actual pipeline published, at their original
 * spacing. Every caption, latency figure and glossary entry on the page came
 * out of a real run. What is simulated is the network, not the inference.
 *
 * This file is hand-written and lives in web/static/. It is the only part of
 * the demo that is not generated from sahaay/ui/.
 */

(function () {
  "use strict";

  const realFetch = window.fetch.bind(window);
  const params = new URLSearchParams(location.search);

  const R = {
    sessions: [],
    lang: params.get("lang"),
    recording: null,
    socket: null,
    timers: [],
    speed: Number(params.get("speed")) || 1,
    autoplay: params.get("play") === "1",
    device: null,
  };

  /* ---------- loading the recording ---------- */

  async function load() {
    if (R.recording) return;

    const index = await (await realFetch("../sessions.json")).json();
    R.sessions = index.sessions || [];
    if (!R.sessions.length) throw new Error("no recordings were published with this site");

    const match = R.sessions.find((s) => s.code === R.lang) || R.sessions[0];
    R.lang = match.code;
    R.recording = await (await realFetch("../" + match.file)).json();

    const first = R.recording.events.find((e) => e.kind === "status" && e.device);
    R.device = first ? first.device : null;
  }

  /* ---------- fetch stub ---------- */

  function json(body) {
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  window.fetch = async function (input, init) {
    const url = String(typeof input === "string" ? input : input.url);

    if (!url.startsWith("/api/")) return realFetch(input, init);

    await load();

    if (url === "/api/status") {
      // running:false so the page opens idle and the visitor presses Start
      // themselves. The recording's own first event says running:true, which
      // is true of the machine that made it, not of this page.
      return json({
        running: false,
        mock: false,
        device: R.device,
        target_language: R.lang,
        languages: R.sessions.map((s) => ({ code: s.code, name: s.name })),
      });
    }

    if (url === "/api/start") {
      play();
      return json({ ok: true, running: true });
    }

    if (url === "/api/stop") {
      halt();
      return json({ ok: true, running: false, saved: true });
    }

    if (url.startsWith("/api/language/")) {
      const code = url.split("/").pop();
      // Each language is a separate recording, so switching means loading a
      // different one. A reload is a far more reliable reset than trying to
      // unwind app.js's caption state from outside it.
      location.search = `?lang=${encodeURIComponent(code)}&play=1&speed=${R.speed}`;
      return json({ ok: true, target_language: code });
    }

    return json({ ok: false, error: "not available in the hosted demo" });
  };

  /* ---------- WebSocket stub ---------- */

  class ReplaySocket {
    constructor() {
      this.readyState = 1;
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      R.socket = this;
      setTimeout(() => this.onopen && this.onopen({}), 0);
    }
    send() {}
    close() {
      this.readyState = 3;
      // Deliberately does not fire onclose: app.js reconnects on close, and
      // a reconnect loop against a socket that does not exist would spin.
    }
  }
  window.WebSocket = ReplaySocket;

  function deliver(event) {
    if (R.socket && R.socket.onmessage) {
      R.socket.onmessage({ data: JSON.stringify(event) });
    }
  }

  /* ---------- the replay itself ---------- */

  function clearTimers() {
    R.timers.forEach(clearTimeout);
    R.timers = [];
  }

  function statusForReplay(event) {
    /* A recording contains the pipeline's whole lifecycle, including the
     * model loading that happened before the session started. Replaying
     * that verbatim makes the UI describe the recording machine instead of
     * the replay: "Loading…" disables the Start button, and the
     * running:false that the summariser publishes flips the button back to
     * "Start" while captions are still arriving.
     *
     * Keep the parts that describe the machine and the run - the device
     * panel, the live metrics, the notes toast - and drop the lifecycle.
     */
    const out = { kind: "status" };
    if (event.device) out.device = event.device;
    if (event.metrics) out.metrics = event.metrics;
    if (event.stage === "summarising") out.stage = "summarising";
    return out.device || out.metrics || out.stage ? out : null;
  }

  function play() {
    clearTimers();
    const events = R.recording.events;
    const speed = R.speed > 0 ? R.speed : 1;

    events.forEach((event) => {
      const payload = event.kind === "status" ? statusForReplay(event) : event;
      if (!payload) return;
      R.timers.push(setTimeout(() => deliver(payload), (event.t * 1000) / speed));
    });

    const end = (events[events.length - 1].t * 1000) / speed;
    R.timers.push(
      setTimeout(() => {
        deliver({ kind: "status", running: false, device: R.device });
      }, end + 400)
    );
  }

  function halt() {
    clearTimers();
  }

  /* ---------- replay speed control ---------- */

  function addSpeedControl() {
    const bar = document.querySelector(".bar-right");
    if (!bar) return;

    const wrap = document.createElement("label");
    wrap.className = "replay-controls";
    wrap.innerHTML = '<span>Replay</span><select id="replay-speed" class="btn"></select>';
    const select = wrap.querySelector("select");
    // Include whatever speed the URL asked for, or the control renders blank
    // for anything outside the list - which is what ?speed=8 did.
    const speeds = [...new Set([1, 2, 4, R.speed])].sort((a, b) => a - b);
    speeds.forEach((x) => {
      const opt = document.createElement("option");
      opt.value = String(x);
      opt.textContent = x + "×";
      select.appendChild(opt);
    });
    select.value = String(R.speed);
    select.onchange = () => {
      location.search = `?lang=${encodeURIComponent(R.lang)}&play=1&speed=${select.value}`;
    };
    bar.insertBefore(wrap, bar.firstChild);
  }

  async function describeOrigin() {
    // The device badge will read "CPU" because that is what this recording
    // ran on, and a visitor could reasonably read that as "so it does not
    // use the NPU after all". Say where it came from, from the recording
    // itself rather than from a hardcoded sentence.
    await load();
    const el = document.getElementById("replay-origin");
    if (!el) return;
    const label = (R.device && (R.device.provider_label || R.device.provider)) || "an unknown provider";
    const session = R.sessions.find((s) => s.code === R.lang);
    // textContent, not innerHTML: this string is assembled from a JSON file,
    // and a page that interpolates data into markup is a habit worth not
    // having even when the data is your own.
    el.textContent = session && session.mode === "mock"
      ? "Recorded in mock mode — the event stream is real, the transcript is scripted. "
      : `Recorded on ${label} — that is what the badge reports. `;
  }

  document.addEventListener("DOMContentLoaded", () => {
    addSpeedControl();
    describeOrigin();
    if (R.autoplay) {
      // boot() is async - it awaits the status request before enabling the
      // button. A fixed delay is a race: too short and the click lands on a
      // disabled button and nothing ever plays. Poll until it is ready.
      const deadline = Date.now() + 10000;
      const tryStart = () => {
        const btn = document.getElementById("toggle");
        if (btn && !btn.disabled) {
          btn.click();
        } else if (Date.now() < deadline) {
          setTimeout(tryStart, 100);
        }
      };
      setTimeout(tryStart, 100);
    }
  });
})();
