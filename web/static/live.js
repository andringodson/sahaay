/* Sahaay, running in the visitor's browser.
 *
 * The desktop app's claim is that a student's lecture never leaves their
 * machine. A hosted demo normally contradicts that: you upload audio to a
 * server and it sends captions back. This page does not. Whisper is fetched
 * once from a CDN and executed here, in the tab, on the visitor's own
 * silicon - WebGPU where the browser has it, WASM where it does not. The
 * audio never goes anywhere.
 *
 * So this is the same argument as the product, on hardware anyone already
 * has, which is why it belongs on the public site while the pipeline itself
 * does not.
 *
 * Like replay.js, it swaps the transport and nothing else: app.js - the
 * product's own UI, copied byte for byte by scripts/build_web.py - boots
 * normally, asks for /api/status, opens a socket, and renders what arrives.
 * What arrives here is produced by real inference a few centimetres away
 * instead of by a recording.
 *
 * What it is NOT: the desktop pipeline. There is no NPU in a browser, no
 * system-audio loopback without the visitor sharing a tab, and the 650 MB
 * translation model is not something to push down a phone connection. The
 * banner says so rather than letting the page imply otherwise.
 */

(function () {
  "use strict";

  const realFetch = window.fetch.bind(window);

  // Pinned rather than floating: a major version of the runtime landing
  // overnight must not be able to break a page someone is presenting from.
  const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";
  const MODEL = "onnx-community/whisper-base";

  const RATE = 16000;
  const FRAME = 512;                 // 32 ms, same frame the product uses
  const SILENCE_FLUSH_MS = 700;      // sahaay/config.py AudioConfig
  const MIN_SEGMENT_S = 1.0;
  const MAX_SEGMENT_S = 12.0;
  const SPEECH_RMS = 0.012;          // energy gate; Silero is not worth 2 MB here

  const L = {
    socket: null,
    ctx: null,
    stream: null,
    node: null,
    transcriber: null,
    backend: "wasm",
    running: false,
    index: 0,
    glossary: null,
    seen: new Set(),
    // Segmentation state
    buffer: [],
    bufferLen: 0,
    silentFor: 0,
    speaking: false,
    queue: [],
    busy: false,
    levelSentAt: 0,
  };

  /* ---------- talking to app.js ---------- */

  function json(body) {
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  function emit(event) {
    if (L.socket && L.socket.onmessage) {
      L.socket.onmessage({ data: JSON.stringify(event) });
    }
  }

  function status(text, tone) {
    const el = document.getElementById("live-status");
    if (el) {
      el.textContent = text ? text + " " : "";
      el.style.color = tone === "bad" ? "var(--danger)" : "";
    }
  }

  function device() {
    // Honest about what is executing. The desktop badge names an ONNX
    // Runtime execution provider; this one names the browser backend, and
    // npu_active stays false because a browser has no Hexagon.
    const label = L.backend === "webgpu" ? "WebGPU (your GPU)" : "WASM (your CPU)";
    return {
      provider: L.backend === "webgpu" ? "WebGPU" : "WASM",
      provider_label: label,
      available_providers: L.backend === "webgpu" ? ["WebGPU", "WASM"] : ["WASM"],
      ort_version: "transformers.js",
      machine: navigator.platform || "browser",
      processor: navigator.userAgent.slice(0, 60),
      is_arm64: /arm|aarch64/i.test(navigator.userAgent),
      npu_active: false,
      fallback_reason:
        "Running in a browser: no Hexagon NPU, and translation needs the desktop app.",
      qnn_hardware: null,
    };
  }

  /* ---------- fetch + WebSocket stubs ---------- */

  window.fetch = async function (input, init) {
    const url = String(typeof input === "string" ? input : input.url);
    if (!url.startsWith("/api/")) return realFetch(input, init);

    if (url === "/api/status") {
      return json({
        running: L.running,
        mock: false,
        device: device(),
        target_language: "en",
        // Only English: translation is a 650 MB model, and offering
        // languages that quietly do nothing would be worse than offering
        // none. The desktop app does the other eight.
        languages: [{ code: "en", name: "English (captions only)" }],
      });
    }
    if (url === "/api/start") {
      try {
        await start();
        return json({ ok: true, running: true });
      } catch (err) {
        status(String((err && err.message) || err), "bad");
        return json({ ok: false, error: String((err && err.message) || err) });
      }
    }
    if (url === "/api/stop") {
      await stop();
      return json({ ok: true, running: false, saved: false });
    }
    if (url.startsWith("/api/language/")) return json({ ok: true });
    return json({ ok: false, error: "not available in the browser build" });
  };

  class LiveSocket {
    constructor() {
      this.readyState = 1;
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      L.socket = this;
      setTimeout(() => this.onopen && this.onopen({}), 0);
    }
    send() {}
    close() {
      this.readyState = 3;
      // No onclose: app.js reconnects on close, and a reconnect loop
      // against a socket that does not exist would spin forever.
    }
  }
  window.WebSocket = LiveSocket;

  /* ---------- the glossary, same vocabulary as the desktop fallback ---------- */

  async function loadGlossary() {
    if (L.glossary) return;
    try {
      L.glossary = await (await realFetch("../static/glossary.json")).json();
    } catch (_) {
      L.glossary = {};
    }
  }

  // Ported from sahaay.llm.seed_entry. A lecture says "diagonalization";
  // the glossary stores "diagonalize", and exact lookup misses it.
  function seedEntry(word) {
    const w = word.toLowerCase().replace(/^-+|-+$/g, "");
    if (L.glossary[w]) return [w, L.glossary[w]];

    const candidates = [];
    for (const suffix of ["s", "es", "ed", "ing"]) {
      if (w.endsWith(suffix)) {
        const stem = w.slice(0, -suffix.length);
        candidates.push(stem, stem + "e");
      }
    }
    for (const [suffix, replacement] of [
      ["ization", "ize"], ["isation", "ise"], ["ation", "ate"],
    ]) {
      if (w.endsWith(suffix)) candidates.push(w.slice(0, -suffix.length) + replacement);
    }
    if (w.endsWith("ices")) candidates.push(w.slice(0, -4) + "ix");

    for (const c of candidates) {
      if (L.glossary[c]) return [c, L.glossary[c]];
    }
    return [w, null];
  }

  function explainJargon(text) {
    for (const word of text.match(/[A-Za-z][A-Za-z-]{3,}/g) || []) {
      const [canonical, explanation] = seedEntry(word);
      if (!explanation || L.seen.has(canonical)) continue;
      L.seen.add(canonical);
      emit({
        kind: "gloss",
        term: canonical,
        explanation,
        language: "en",
        source_line: text,
        latency_ms: 0,
        backend: "seeded vocabulary (in-browser)",
      });
    }
  }

  /* ---------- audio ---------- */

  async function openStream(sourceKind) {
    if (sourceKind === "tab") {
      if (!navigator.mediaDevices.getDisplayMedia) {
        throw new Error("This browser cannot share tab audio. Use the microphone.");
      }
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false },
      });
      if (!stream.getAudioTracks().length) {
        stream.getTracks().forEach((t) => t.stop());
        throw new Error(
          'No audio in that share. Re-pick the tab and tick "Also share tab audio".'
        );
      }
      // The video track is only the price of admission for tab audio.
      stream.getVideoTracks().forEach((t) => t.stop());
      return stream;
    }
    return navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  }

  function sourceKind() {
    const select = document.getElementById("live-source-select");
    return select ? select.value : "mic";
  }

  /* ---------- segmentation: the product's rules, in JS ---------- */

  function onFrame(frame) {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);

    const now = performance.now();
    if (now - L.levelSentAt >= 200) {
      L.levelSentAt = now;
      emit({ kind: "level", rms: Number(rms.toFixed(5)) });
    }

    const loud = rms > SPEECH_RMS;
    if (loud) {
      L.speaking = true;
      L.silentFor = 0;
    } else if (L.speaking) {
      L.silentFor += (frame.length / RATE) * 1000;
    }

    if (L.speaking) {
      L.buffer.push(frame);
      L.bufferLen += frame.length;
    }

    const longEnough = L.bufferLen >= MIN_SEGMENT_S * RATE;
    const paused = L.silentFor >= SILENCE_FLUSH_MS;
    const full = L.bufferLen >= MAX_SEGMENT_S * RATE;

    if (L.speaking && ((paused && longEnough) || full)) flush();
    else if (paused && !longEnough) reset();
  }

  function reset() {
    L.buffer = [];
    L.bufferLen = 0;
    L.silentFor = 0;
    L.speaking = false;
  }

  function flush() {
    const audio = new Float32Array(L.bufferLen);
    let at = 0;
    for (const frame of L.buffer) {
      audio.set(frame, at);
      at += frame.length;
    }
    reset();
    L.queue.push(audio);
    drain();
  }

  async function drain() {
    if (L.busy || !L.queue.length) return;
    L.busy = true;
    // Newest first would reorder the transcript; a live captioner stays in
    // order and drops the oldest under pressure, the way the pipeline does.
    while (L.queue.length > 3) L.queue.shift();

    const audio = L.queue.shift();
    const index = L.index++;
    emit({ kind: "partial", text: "…", index });

    const started = performance.now();
    try {
      const out = await L.transcriber(audio, {
        chunk_length_s: 30,
        return_timestamps: false,
      });
      const text = (out && out.text ? out.text : "").trim();
      const ms = performance.now() - started;
      const seconds = audio.length / RATE;

      if (text && !/^[\s.]*$/.test(text)) {
        emit({
          kind: "caption",
          index,
          text,
          language: "en",
          start_s: 0,
          latency_ms: Math.round(ms),
          rtf: Number((ms / 1000 / seconds).toFixed(3)),
          provider: device().provider,
        });
        emit({
          kind: "metric",
          uptime_s: 0,
          audio_seconds: seconds,
          realtime_factor: Number((ms / 1000 / seconds).toFixed(3)),
          stages: [],
        });
        await loadGlossary();
        explainJargon(text);
      }
    } catch (err) {
      emit({ kind: "error", message: "Transcription failed: " + err });
    } finally {
      L.busy = false;
      if (L.queue.length) drain();
    }
  }

  /* ---------- lifecycle ---------- */

  async function webgpuAvailable() {
    if (!navigator.gpu) return false;
    try {
      return Boolean(await navigator.gpu.requestAdapter());
    } catch (_) {
      return false;
    }
  }

  async function loadModel() {
    if (L.transcriber) return;

    status("fetching Whisper (about 80 MB, cached after this)…");
    const progress = document.getElementById("live-bar");

    const { pipeline, env } = await import(/* @vite-ignore */ TRANSFORMERS);
    env.allowLocalModels = false;

    const onProgress = (p) => {
      if (progress && p && p.status === "progress" && p.total) {
        progress.style.width = Math.round((p.loaded / p.total) * 100) + "%";
      }
    };

    // WebGPU where it exists, WASM where it does not. The badge says which,
    // because a demo that hides what it ran on is the thing this project
    // keeps arguing against.
    //
    // Decide by asking for an adapter, not by try/catch around pipeline().
    // `navigator.gpu` is present in plenty of browsers that then fail to
    // return an adapter - headless Chromium is one - and by the time that
    // surfaces the runtime has already fixed its backend list, so the
    // "fallback" asks for WASM and is told "no available backend found
    // ERR: [webgpu]". Probing first means one decision, made correctly.
    L.backend = (await webgpuAvailable()) ? "webgpu" : "wasm";

    // Only the precision falls back. q8 is the safe floor on both.
    const dtypes = L.backend === "webgpu" ? ["q4", "q8"] : ["q8", "fp32"];
    let failure = null;
    for (const dtype of dtypes) {
      try {
        L.transcriber = await pipeline("automatic-speech-recognition", MODEL, {
          device: L.backend, dtype, progress_callback: onProgress,
        });
        break;
      } catch (err) {
        failure = err;
        console.warn(`${L.backend}/${dtype} did not load`, err);
      }
    }
    if (!L.transcriber) throw failure || new Error("Whisper would not load");

    if (progress) progress.style.width = "100%";
    status("warming up…");
    await L.transcriber(new Float32Array(RATE));   // pay graph setup before the lecture
  }

  async function start() {
    await loadModel();

    L.stream = await openStream(sourceKind());
    L.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RATE });
    const source = L.ctx.createMediaStreamSource(L.stream);

    // Capture on the audio thread. Whisper blocks the main thread for
    // seconds at a time, and a ScriptProcessorNode - which delivers on the
    // main thread - simply stops recording while that happens: measured as
    // whole sentences missing from the transcript.
    let captured = false;
    if (L.ctx.audioWorklet) {
      try {
        await L.ctx.audioWorklet.addModule("../static/capture-worklet.js");
        L.node = new AudioWorkletNode(L.ctx, "sahaay-capture");
        L.node.port.onmessage = (e) => {
          if (L.running) onFrame(e.data);
        };
        captured = true;
      } catch (err) {
        console.warn("AudioWorklet unavailable, falling back", err);
      }
    }
    if (!captured) {
      // Older Safari. Loses audio under load, but captions beat nothing.
      L.node = L.ctx.createScriptProcessor(4096, 1, 1);
      L.node.onaudioprocess = (e) => {
        if (!L.running) return;
        const input = e.inputBuffer.getChannelData(0);
        for (let at = 0; at + FRAME <= input.length; at += FRAME) {
          onFrame(input.slice(at, at + FRAME));
        }
      };
    }

    source.connect(L.node);
    // Destination gain of zero: the node needs a sink, and routing the
    // microphone back to the speakers would make the room howl.
    const mute = L.ctx.createGain();
    mute.gain.value = 0;
    L.node.connect(mute);
    mute.connect(L.ctx.destination);

    L.running = true;
    reset();
    status("listening");
    emit({ kind: "status", running: true, device: device() });
  }

  async function stop() {
    L.running = false;
    status("");
    if (L.node) L.node.disconnect();
    if (L.ctx) await L.ctx.close().catch(() => {});
    if (L.stream) L.stream.getTracks().forEach((t) => t.stop());
    L.node = L.ctx = L.stream = null;
    emit({ kind: "status", running: false, device: device() });
  }

  /* ---------- injected controls ---------- */

  function addSourcePicker() {
    const bar = document.querySelector(".bar-right");
    if (!bar) return;
    const wrap = document.createElement("label");
    wrap.className = "live-source";
    wrap.innerHTML =
      "<span>Listen to</span>" +
      '<select id="live-source-select">' +
      '<option value="mic">Microphone</option>' +
      '<option value="tab">A browser tab</option>' +
      "</select>";
    bar.insertBefore(wrap, bar.firstChild);

    const banner = document.querySelector(".live-banner");
    if (banner) {
      const bar2 = document.createElement("div");
      bar2.className = "live-progress";
      bar2.innerHTML = '<span id="live-bar"></span>';
      banner.appendChild(bar2);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    addSourcePicker();
    loadGlossary();

    const empty = document.getElementById("empty");
    if (empty) {
      empty.innerHTML =
        "Press <kbd>Start</kbd> and talk, or pick <b>A browser tab</b> and share a " +
        "tab playing a lecture. The first run downloads Whisper (about 80 MB) and " +
        "then runs it here &mdash; your audio never leaves this tab.";
    }
  });
})();
