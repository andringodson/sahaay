/* Sahaay background matrix.

   A field of very dark glyphs behind the interface that wakes up around the
   cursor and sends a ripple out from every click or tap.

   The budget matters more than the look. On /live Whisper runs in the same
   browser, and the real-time factor is the number this whole project is
   about, so a background that costs a few milliseconds a frame would be
   measurably making the product worse. Hence:

   - The dim field is drawn once per resize into an offscreen canvas and
     blitted; only "hot" cells (near the cursor, on a ripple, under a falling
     drop) are drawn individually, from a pre-rendered glyph atlas.
   - With nothing moving, the loop stops. No timer, no requestAnimationFrame:
     an idle page costs exactly nothing. Ambient rain (data-ambient="on") is
     for the landing page only, where nothing else competes for the CPU, and
     it runs at 30 fps.
   - Hidden tab: stopped. prefers-reduced-motion: the static field only.
   - Device pixel ratio is capped at 1.5; glyphs this dim gain nothing from
     4K backing stores, and fill cost scales with it.

   window.SahaayMatrix.stats exposes frame counts and draw time so the cost
   can be measured rather than asserted (scripts/check_live.py reads it). */

(() => {
  "use strict";

  const script = document.currentScript;
  const AMBIENT = !!(script && script.dataset.ambient === "on");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  const CELL = 22;                 // grid pitch, CSS px
  const FONT_PX = 13;
  const COLOR = "#4db8ff";         // the accent; every shade is this at low alpha
  const BASE_ALPHA = 0.075;        // the resting field: barely there on OLED black
  const HOT_ALPHA = 0.5;           // the brightest a cell ever gets
  const RADIUS = 150;              // cursor influence, CSS px
  const IDLE_MS = 1400;            // cursor glow fades out after this long still
  const MAX_RIPPLES = 4;

  // Latin digits, maths, and Devanagari consonants: the lecture material the
  // app is for, and the script its name is written in.
  const GLYPHS = Array.from("01234567890λπΣΔ∫√∞≈∂θ" +
    "कखगघचजटडतथदधनपफबभमयरलवशसह");

  const canvas = document.createElement("canvas");
  canvas.className = "matrix-bg";
  canvas.setAttribute("aria-hidden", "true");
  const ctx = canvas.getContext("2d", { alpha: true });

  const base = document.createElement("canvas");
  const bctx = base.getContext("2d");
  const atlas = document.createElement("canvas");

  let dpr = 1, w = 0, h = 0, cols = 0, rows = 0;
  let glyph = new Uint8Array(0);
  let heat = new Float32Array(0);
  let isHot = new Uint8Array(0);
  let hot = [];
  let drops = [];
  const ripples = [];

  const pointer = { x: -1e4, y: -1e4, on: false, strength: 0, movedAt: 0 };
  let running = false, last = 0, lastDraw = 0;

  const stats = { frames: 0, drawMs: 0, maxMs: 0, running: () => running };
  window.SahaayMatrix = { stats };

  function buildAtlas() {
    const s = Math.ceil(CELL * dpr);
    atlas.width = s * GLYPHS.length;
    atlas.height = s;
    const a = atlas.getContext("2d");
    a.clearRect(0, 0, atlas.width, atlas.height);
    a.fillStyle = COLOR;
    a.font = `${FONT_PX * dpr}px "Cascadia Code", Consolas, "Nirmala UI", "Noto Sans Devanagari", monospace`;
    a.textAlign = "center";
    a.textBaseline = "middle";
    GLYPHS.forEach((g, i) => a.fillText(g, i * s + s / 2, s / 2));
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = base.width = Math.round(w * dpr);
    canvas.height = base.height = Math.round(h * dpr);
    cols = Math.ceil(w / CELL);
    rows = Math.ceil(h / CELL);

    const n = cols * rows;
    glyph = new Uint8Array(n);
    for (let i = 0; i < n; i++) glyph[i] = (Math.random() * GLYPHS.length) | 0;
    heat = new Float32Array(n);
    isHot = new Uint8Array(n);
    hot = [];

    buildAtlas();
    bctx.clearRect(0, 0, base.width, base.height);
    bctx.globalAlpha = BASE_ALPHA;
    for (let i = 0; i < n; i++) blit(bctx, i);
    bctx.globalAlpha = 1;

    drops = [];
    if (AMBIENT) {
      const count = Math.max(6, Math.round(cols / 5));
      for (let k = 0; k < count; k++) drops.push(newDrop(true));
    }
    paint();
  }

  function blit(c, i) {
    const s = Math.ceil(CELL * dpr);
    const x = (i % cols) * CELL * dpr;
    const y = ((i / cols) | 0) * CELL * dpr;
    c.drawImage(atlas, glyph[i] * s, 0, s, s, x, y, s, s);
  }

  function bump(i, v) {
    if (v > heat[i]) heat[i] = v;
    if (!isHot[i]) { isHot[i] = 1; hot.push(i); }
  }

  function newDrop(scatter) {
    return {
      col: (Math.random() * cols) | 0,
      y: scatter ? Math.random() * rows : -Math.random() * rows * 0.6,
      speed: 4 + Math.random() * 7,          // cells per second
    };
  }

  function step(dt) {
    // Cool everything that is lit; forget what has gone dark.
    const keep = Math.pow(0.9, dt / 16.7);
    let j = 0;
    for (let k = 0; k < hot.length; k++) {
      const i = hot[k];
      heat[i] *= keep;
      if (heat[i] > 0.015) hot[j++] = i;
      else { heat[i] = 0; isHot[i] = 0; }
    }
    hot.length = j;

    // Cursor: full strength while moving, fading once it rests.
    const now = performance.now();
    const target = pointer.on && now - pointer.movedAt < IDLE_MS ? 1 : 0;
    pointer.strength += (target - pointer.strength) * Math.min(1, dt / 180);
    if (pointer.strength > 0.02) {
      const c0 = Math.max(0, ((pointer.x - RADIUS) / CELL) | 0);
      const c1 = Math.min(cols - 1, ((pointer.x + RADIUS) / CELL) | 0);
      const r0 = Math.max(0, ((pointer.y - RADIUS) / CELL) | 0);
      const r1 = Math.min(rows - 1, ((pointer.y + RADIUS) / CELL) | 0);
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const dx = c * CELL + CELL / 2 - pointer.x;
          const dy = r * CELL + CELL / 2 - pointer.y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < RADIUS) {
            const f = 1 - d / RADIUS;
            bump(r * cols + c, f * f * pointer.strength);
          }
        }
      }
    }

    // Ripples: a ring of light expanding from each click.
    for (let k = ripples.length - 1; k >= 0; k--) {
      const rp = ripples[k];
      rp.r += dt * 0.55;
      rp.life = 1 - rp.r / rp.max;
      if (rp.life <= 0) { ripples.splice(k, 1); continue; }
      const c0 = Math.max(0, ((rp.x - rp.r - CELL) / CELL) | 0);
      const c1 = Math.min(cols - 1, ((rp.x + rp.r + CELL) / CELL) | 0);
      const r0 = Math.max(0, ((rp.y - rp.r - CELL) / CELL) | 0);
      const r1 = Math.min(rows - 1, ((rp.y + rp.r + CELL) / CELL) | 0);
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const dx = c * CELL + CELL / 2 - rp.x;
          const dy = r * CELL + CELL / 2 - rp.y;
          const band = Math.abs(Math.sqrt(dx * dx + dy * dy) - rp.r);
          if (band < CELL) bump(r * cols + c, (1 - band / CELL) * rp.life * 0.9);
        }
      }
    }

    // Ambient rain.
    for (let k = 0; k < drops.length; k++) {
      const dp = drops[k];
      const before = dp.y | 0;
      dp.y += dp.speed * dt / 1000;
      const after = dp.y | 0;
      if (after !== before && after >= 0 && after < rows) {
        const i = after * cols + dp.col;
        glyph[i] = (Math.random() * GLYPHS.length) | 0;
        bump(i, 0.42);
      }
      if (dp.y >= rows) drops[k] = newDrop(false);
    }

    // A little shimmer in whatever is lit.
    if (hot.length) {
      const i = hot[(Math.random() * hot.length) | 0];
      glyph[i] = (Math.random() * GLYPHS.length) | 0;
    }
  }

  function paint() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(base, 0, 0);
    for (let k = 0; k < hot.length; k++) {
      const i = hot[k];
      ctx.globalAlpha = Math.min(1, heat[i]) * HOT_ALPHA;
      blit(ctx, i);
    }
    ctx.globalAlpha = 1;
  }

  function busy() {
    return hot.length > 0 || ripples.length > 0 || pointer.strength > 0.02 ||
      (pointer.on && performance.now() - pointer.movedAt < IDLE_MS) ||
      (AMBIENT && drops.length > 0);
  }

  function frame(t) {
    if (!running) return;
    const dt = Math.min(64, t - (last || t));
    last = t;

    // Rain alone does not need 60 fps; anything the user did does.
    const interactive = ripples.length > 0 || pointer.strength > 0.02;
    if (!interactive && t - lastDraw < 33) { requestAnimationFrame(frame); return; }
    const dtStep = Math.min(64, t - (lastDraw || t));
    lastDraw = t;

    const t0 = performance.now();
    step(interactive ? dt : dtStep);
    paint();
    const ms = performance.now() - t0;
    stats.frames++;
    stats.drawMs += ms;
    if (ms > stats.maxMs) stats.maxMs = ms;

    if (busy()) requestAnimationFrame(frame);
    else { running = false; last = 0; lastDraw = 0; }
  }

  function wake() {
    if (running || reduced.matches || document.hidden) return;
    running = true;
    last = 0;
    lastDraw = 0;
    requestAnimationFrame(frame);
  }

  window.addEventListener("pointermove", (e) => {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    pointer.on = true;
    pointer.movedAt = performance.now();
    wake();
  }, { passive: true });

  window.addEventListener("pointerdown", (e) => {
    if (ripples.length >= MAX_RIPPLES) ripples.shift();
    ripples.push({ x: e.clientX, y: e.clientY, r: 0, life: 1,
                   max: Math.min(520, Math.max(w, h) * 0.55) });
    wake();
  }, { passive: true });

  document.addEventListener("pointerleave", () => { pointer.on = false; });
  document.documentElement.addEventListener("mouseleave", () => { pointer.on = false; });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) running = false;
    else if (AMBIENT) wake();
  });

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 120);
  });

  const onMotionPref = () => {
    if (reduced.matches) { running = false; hot = []; ripples.length = 0; resize(); }
    else if (AMBIENT) wake();
  };
  if (reduced.addEventListener) reduced.addEventListener("change", onMotionPref);

  function start() {
    document.body.prepend(canvas);
    resize();
    if (AMBIENT) wake();
  }

  if (document.body) start();
  else document.addEventListener("DOMContentLoaded", start, { once: true });
})();
