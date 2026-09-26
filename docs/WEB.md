# The hosted site

**[sahaay-offline.vercel.app](https://sahaay-offline.vercel.app)**

## What is hosted, and what is not

Three things live at that URL, and they are deliberately different:

| | What runs | Where the inference happens |
|---|---|---|
| `/` | the evidence, with its source links | nothing |
| `/live/` | **Whisper, for real** | your browser, on your machine |
| `/demo/` | a recording of a desktop session | nothing; it is a replay |

**The desktop pipeline is not hosted, and cannot be.** That is not a
limitation to work around — it is the product:

- **There is no NPU in a datacenter.** The entire argument of this project is
  that the Hexagon NPU is what makes two models run side by side without the
  captions suffering. A cloud deployment would run on a rented x86 CPU and
  demonstrate the opposite.
- **There is no audio.** Capture is WASAPI loopback — whatever is playing on
  *the user's* speakers. A server has nothing to listen to.
- **The models are gigabytes.** Whisper Small, NLLB-200 and Llama 3.2 together
  are well past any serverless bundle limit, and cold-starting them per
  request would make live captioning meaningless.
- **Most importantly, it would invert the claim.** Sahaay exists because a
  student's lectures should not leave their machine. Uploading their audio to
  a server to prove that would be an odd way to make the point.

## The browser build

`/live/` resolves that tension rather than dodging it. The page is static
files; **Whisper is fetched once from a CDN and executed in the visitor's own
browser** — WebGPU where the browser has an adapter, WebAssembly where it does
not. The audio never leaves the tab. That is the same claim the desktop app
makes, on hardware every visitor already has, which is why it belongs on a
public URL while the pipeline itself does not.

`web/static/live.js` swaps the transport and nothing else, exactly as
`replay.js` does: `app.js` — the product's own UI — boots, asks for
`/api/status`, opens a socket, and renders what arrives. What arrives is
produced by real inference a few centimetres away.

The jargon sidebar uses the same seeded vocabulary the desktop app falls back
to when no language model is installed. `build_web.py` exports it to
`web/static/glossary.json` from `sahaay.llm.SEED_GLOSSARY`, and a test asserts
the two match, so they cannot drift.

**What it is not.** Whisper Base rather than Small; a shared browser tab
rather than all system audio; no translation, because NLLB-200 is 650 MB and
that is not something to push down a phone connection; and no NPU, so it
cannot show the measurement the project is actually about. The page says all
of this in its own banner rather than letting a visitor infer otherwise.

### Two things a real browser found

Neither was visible to the test suite, and both were found by
`scripts/check_live.py`, which drives the page in Chromium with its fake
capture device fed from `testaudio/lecture_long.wav`.

**The WebGPU probe.** The first version tried `pipeline(..., device: "webgpu")`
inside a `try`, and fell back to WASM in the `catch`. `navigator.gpu` exists in
plenty of browsers that then fail to return an adapter — headless Chromium is
one — and by the time that surfaced the runtime had already fixed its backend
list, so the fallback asked for WASM and was told *"no available backend found
ERR: [webgpu]"*. The page loaded perfectly and never captioned. Asking
`requestAdapter()` first makes it one decision, made correctly.

**Capture on the wrong thread.** Whisper runs as WebAssembly on the main
thread and blocks it for seconds. A `ScriptProcessorNode` delivers its
callbacks on that same thread, so while one segment was being transcribed the
next was not being recorded — measured as whole sentences missing. Moving
capture into an `AudioWorkletProcessor`, which runs on the audio thread,
recovered them: the same clip went from two captions with the second sentence
lost and no glossary entries, to two captions in order with both terms
explained. Under load the captions now arrive late rather than the audio
disappearing, which is the same trade the desktop pipeline makes.

## How the demo works

The demo is not a mockup and not a video. It is the product's own UI —
`sahaay/ui/app.js` and `sahaay/ui/style.css`, copied byte for byte — with one
file added underneath it.

`web/static/replay.js` stubs `fetch` and `WebSocket`. That is all it does.
`app.js` boots exactly as it does against the local server, asks for
`/api/status`, opens a socket, and starts rendering whatever arrives. What
arrives is a recording made by `scripts/record_demo.py`: the events the real
pipeline really published, at their original spacing.

**Every caption, translation, latency figure and glossary entry on that page
came out of a real run** on real weights — Whisper Small transcribing,
NLLB-200 translating, on CPU. What is simulated is the network, not the
inference. The device badge shows the provider that actually ran, which is
`CPUExecutionProvider` and says so.

### Why replay rather than reimplement

A hand-written demo drifts. Someone fixes a rendering bug in the product and
the public demo keeps the bug while claiming to be the application.
`scripts/build_web.py` generates `web/` from `sahaay/ui/`, and
`tests/test_web_build.py` fails if the copies diverge — so a stale site is a
red CI run rather than something a judge discovers.

## The background, and what it is allowed to cost

Every page sits on OLED black with a field of very dark glyphs behind it
(`sahaay/ui/matrix.js`, copied to the site like the rest of the UI). It lights
up around the cursor and sends a ripple from each click or tap.

On `/live/` that canvas shares a browser with Whisper, and the real-time
factor is the number this project is about, so it runs on a strict budget:

- The dim field is drawn once per resize and blitted; only lit cells are
  drawn per frame, from a pre-rendered glyph atlas. Measured in headless
  Chromium at 1440×900: **0.6–0.7 ms per frame** while the cursor moves,
  under 2 ms at worst.
- **With nothing moving, the loop stops**: no timer and no
  `requestAnimationFrame`, so an idle page, and a lecture being captioned
  while you read, costs nothing. The ambient rain runs only on the landing
  page (`data-ambient="on"`), at 30 fps, where nothing else needs the CPU.
- A hidden tab stops it. `prefers-reduced-motion` gets the static field
  only. Device pixel ratio is capped at 1.5.

`window.SahaayMatrix.stats` reports frame count and draw time, so that is
measured, not assumed.

## Rebuilding it

```powershell
python scripts/make_lecture_audio.py                      # synthesise the lecture
python scripts/record_demo.py --real --audio-file testaudio/lecture_long.wav `
                              --language hi --out web/session.hi.json
python scripts/build_web.py                               # assemble web/
python scripts/shoot_web.py                               # render it and check
```

`record_demo.py` without `--real` uses the scripted mock transcript, so the
site can be rebuilt on a machine with no weights downloaded — the event
stream is real, the words are not, and the recording says which it is in its
`mode` field.

One recording per language; `build_web.py` scans for `web/session.*.json` and
writes the picker's index. The language selector on the demo switches between
recordings.

## What the recording is checked for

`tests/test_web_build.py` asserts a recording is a session and not just
well-formed JSON: captions present, timestamps monotonic and non-zero, every
caption carrying text and an index. It also asserts **no local paths** ship in
the payload — the session-notes event carries the absolute path a transcript
was saved to, which on a development machine is a home directory. That check
was not hypothetical; it caught exactly that leak the first time it ran.

## Deployment

Static files, no build step, no serverless functions:

```powershell
vercel --prod
```

`vercel.json` sets `outputDirectory: web` and a content-security policy. It
is no longer a flat deny: `/live/` has to reach jsdelivr for the runtime and
huggingface.co for the weights, and needs `wasm-unsafe-eval` to compile
them. Everything else stays shut, and `connect-src` still forbids the page
from posting anywhere. `.vercelignore` keeps
`models/` — over a gigabyte — out of the upload.

The project is connected to the GitHub repository, so a push to `main`
deploys on its own; the command above is for deploying without one.

**On the name.** `sahaay.vercel.app` was already taken by someone else, so
the deployment is `sahaay-offline.vercel.app`. Worth knowing before putting a
URL on a slide — check the alias the deploy actually returns rather than the
one you expected.

`scripts/shoot_web.py --base <url>` runs the same browser checks against the
deployment rather than a local copy — the headers, the rewrite rules and the
CSP only exist on the real host. One caveat: a push deploys, and the alias
moves to the new deployment. Running the check against the live URL while a
deploy is in flight fails in confusing ways (captions that never arrive, a
Start button that never flips). Let the deploy settle first.

The irony is deliberate and worth stating out loud: the site that explains why
this application must run locally is the only part of it that runs anywhere
else.
