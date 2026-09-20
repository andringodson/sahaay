# The hosted site

**[sahaay-offline.vercel.app](https://sahaay-offline.vercel.app)**

## What is hosted, and what is not

Sahaay itself is not hosted, and cannot be. That is not a limitation to work
around — it is the product:

- **There is no NPU in a datacenter.** The entire argument of this project is
  that the Hexagon NPU is what makes two models run side by side without the
  captions suffering. A cloud deployment would run on a rented x86 CPU and
  demonstrate the opposite.
- **There is no audio.** Capture is WASAPI loopback — whatever is playing on
  *the user's* speakers. A server has nothing to listen to.
- **The models are gigabytes.** Whisper, NLLB-200 and Llama 3.2 together are
  well past any serverless bundle limit, and cold-starting them per request
  would make live captioning meaningless.
- **Most importantly, it would invert the claim.** Sahaay exists because a
  student's lectures should not leave their machine. Uploading their audio to
  a server to prove that would be an odd way to make the point.

So what is hosted is the **evidence and a replay**: the measured numbers with
their source links, and the real interface replaying a real session.

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

`vercel.json` sets `outputDirectory: web` and a content-security policy that
forbids the page from talking to anything but itself. `.vercelignore` keeps
`models/` — over a gigabyte — out of the upload.

The project is connected to the GitHub repository, so a push to `main`
deploys on its own; the command above is for deploying without one.

**On the name.** `sahaay.vercel.app` was already taken by someone else, so
the deployment is `sahaay-offline.vercel.app`. Worth knowing before putting a
URL on a slide — check the alias the deploy actually returns rather than the
one you expected.

The irony is deliberate and worth stating out loud: the site that explains why
this application must run locally is the only part of it that runs anywhere
else.
