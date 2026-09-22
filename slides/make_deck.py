"""Write the presentation deck.

The 10-slide outline in docs/PRESENTATION.md is the plan; this is the deck
built from it, with the measured numbers taken from the same docs the README
cites. It is here rather than only in a slide tool for one reason: when a
number changes, the slides have to change with it, and a deck nobody can
regenerate quietly goes stale. The concurrency figure has already reversed
once.

    python slides/make_deck.py

Writes slides/project/, which is the content of the deck at
https://claude.ai/artifact/TpKeZmMe3ipSqivKeph9mG - one HTML file per slide
plus an index. Speaker notes are the <aside> at the end of each slide.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SLIDES = ROOT / "project" / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)

# Palette: the product's own tokens, so the deck and the app look like one thing.
DARK = "#0D1014"
DARK_SURFACE = "#151A21"
DARK_LINE = "#262F3A"
LIGHT = "#F6F8FB"
LIGHT_SURFACE = "#FFFFFF"
LIGHT_LINE = "#D7DFE8"
ON_DARK = "#E8EDF3"
ON_DARK_DIM = "#8D9AAB"
ON_LIGHT = "#131920"
ON_LIGHT_DIM = "#5A6775"
ACCENT_D = "#4DB8FF"
ACCENT_L = "#0B6FB8"
NPU_D = "#00D68F"
NPU_L = "#00845C"
DANGER_D = "#FF5C5C"
DANGER_L = "#C62828"
WARN_L = "#9A6200"

HEAD = "'Rubik', Verdana, sans-serif"
BODY = "'IBM Plex Sans', Verdana, sans-serif"

deck_dark = (
    f"background:{DARK}; color:{ON_DARK}; font-family:{BODY}; "
    "display:flex; flex-direction:column; padding:128px 128px 160px"
)
deck_light = (
    f"background:{LIGHT}; color:{ON_LIGHT}; font-family:{BODY}; "
    "display:flex; flex-direction:column; padding:128px 128px 160px"
)


def footer(text, color):
    return (
        f'<p style="position:absolute; left:128px; bottom:64px; width:1664px; '
        f'font-size:24px; color:{color}">{text}</p>'
    )


SLIDE_FILES = {}


def slide(sid, html):
    SLIDE_FILES[sid] = html.strip() + "\n"


# ---------------------------------------------------------------- 1. cover
slide("cover", f"""
<section id="cover" data-transition="fade" style="{deck_dark}; justify-content:center; gap:32px">
  <p style="font-size:24px; color:{ACCENT_D}; letter-spacing:2px; text-transform:uppercase">Snapdragon&#174; AI Lab Build &amp; Present Challenge 2026</p>
  <h1 style="font-family:{HEAD}; font-size:150px; font-weight:600; line-height:1.0">Sahaay</h1>
  <h2 style="font-family:{HEAD}; font-size:56px; font-weight:400; line-height:1.2; color:{ON_DARK}; width:1400px">The lecture understands you &mdash; offline</h2>
  <p style="font-size:32px; color:{ON_DARK_DIM}; width:1280px; line-height:1.5">Live captions, Indian-language translation and a jargon glossary, running entirely on a Snapdragon-powered PC with the network switched off.</p>
  {footer('sahaay-offline.vercel.app  &#183;  github.com/andringodson/sahaay', ON_DARK_DIM)}
  <aside>Open with the name and one sentence, not the tech stack. Leave the URL on screen: a judge who wants to check a number can, mid-talk.</aside>
</section>
""")

# -------------------------------------------------------------- 2. problem
slide("problem", f"""
<section id="problem" style="{deck_light}; gap:48px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">An engineering lecture in India is rarely in one language</h2>
  <div style="background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:20px; padding:56px; display:flex; flex-direction:column; gap:16px">
    <p style="font-size:44px; line-height:1.4; color:{ON_LIGHT}">&ldquo;Matrix A ka <b style="color:{ACCENT_L}">determinant</b> zero hoga, tabhi non-trivial <b style="color:{ACCENT_L}">solution</b> milega.&rdquo;</p>
  </div>
  <p style="font-size:32px; line-height:1.5; color:{ON_LIGHT_DIM}; width:1600px">A student following that in their second language has two problems at once. They must parse a Hindi sentence <i>and</i> hold on to the English technical terms &mdash; the exact terms that appear in the textbook and the exam.</p>
  <p style="font-size:32px; line-height:1.5; color:{ON_LIGHT_DIM}; width:1600px">Cloud captioning handles code-mixing badly, costs money per minute, and needs connectivity a lecture hall, a hostel or a rural college often does not have.</p>
  {footer('The problem', ON_LIGHT_DIM)}
  <aside>Read the sentence aloud. It does the work: everyone in the room recognises it instantly.</aside>
</section>
""")

# ------------------------------------------------------------------ 3. who
slide("who", f"""
<section id="who" style="{deck_light}; gap:56px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">Who this is for</h2>
  <div style="display:flex; gap:32px">
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:16px">
      <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2; color:{ACCENT_L}">Deaf and hard of hearing</h3>
      <p style="font-size:28px; line-height:1.5; color:{ON_LIGHT_DIM}">A cloud round-trip is too slow to follow a sentence in progress.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:16px">
      <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2; color:{ACCENT_L}">Studying in a second language</h3>
      <p style="font-size:28px; line-height:1.5; color:{ON_LIGHT_DIM}">The lecture is in English; the thinking is in Hindi, Tamil, Bengali.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:16px">
      <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2; color:{ACCENT_L}">No usable connectivity</h3>
      <p style="font-size:28px; line-height:1.5; color:{ON_LIGHT_DIM}">A hall, a hostel, a bus. Bandwidth is not a given.</p>
    </div>
  </div>
  <p style="font-size:36px; line-height:1.5; color:{ON_LIGHT}; width:1600px">Not a hypothetical persona. This is the majority experience in Indian engineering education.</p>
  {footer('Who this is for', ON_LIGHT_DIM)}
  <aside>Do not linger. One breath per card, then the last line slowly.</aside>
</section>
""")

# ----------------------------------------------------------------- 4. demo
slide("demo", f"""
<section id="demo" data-transition="push" style="background:{ACCENT_L}; color:#FFFFFF; font-family:{BODY}; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:128px; gap:32px">
  <h1 style="font-family:{HEAD}; font-size:120px; font-weight:600; line-height:1.05; text-align:center">Demo</h1>
  <p style="font-size:40px; line-height:1.5; color:#FFFFFF; width:1300px; text-align:center">A lecture plays. Captions appear in Hindi. The jargon explains itself in the sidebar.</p>
  <p style="font-size:36px; line-height:1.5; color:#FFFFFF; width:1300px; text-align:center">Then I turn the Wi-Fi off, and nothing changes.</p>
  <aside>Turn the Wi-Fi off on camera. It is the strongest fifteen seconds you have - do not cut it, and do not explain it first. Fallback if anything misbehaves: the recorded replay at sahaay-offline.vercel.app/demo</aside>
</section>
""")

# ------------------------------------------------------------------ 5. npu
slide("npu", f"""
<section id="npu" style="{deck_dark}; gap:40px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">Why this needs an NPU</h2>
  <p style="font-size:32px; line-height:1.5; color:{ON_DARK_DIM}; width:1600px">Sahaay runs two models at once for the length of a lecture: Whisper transcribing, Llama writing glossary entries. On one set of CPU cores they compete, and the captions are what lose.</p>
  <table style="font-family:{BODY}; font-size:32px; color:{ON_DARK}">
    <tr>
      <th style="width:40%; text-align:left; font-size:24px; color:{ON_DARK_DIM}; font-weight:600; padding:16px">WHISPER LATENCY</th>
      <th style="width:20%; text-align:right; font-size:24px; color:{ON_DARK_DIM}; font-weight:600">MEAN</th>
      <th style="width:20%; text-align:right; font-size:24px; color:{ON_DARK_DIM}; font-weight:600">P95</th>
      <th style="width:20%; text-align:right; font-size:24px; color:{ON_DARK_DIM}; font-weight:600">REAL-TIME FACTOR</th>
    </tr>
    <tr>
      <td style="text-align:left">Glossary idle</td>
      <td style="text-align:right">3276 ms</td>
      <td style="text-align:right">3422 ms</td>
      <td style="text-align:right; color:{NPU_D}">0.41</td>
    </tr>
    <tr>
      <td style="text-align:left">Glossary running</td>
      <td style="text-align:right; color:{DANGER_D}">12400 ms</td>
      <td style="text-align:right; color:{DANGER_D}">13242 ms</td>
      <td style="text-align:right; color:{DANGER_D}">1.55</td>
    </tr>
  </table>
  <div style="background:{DARK_SURFACE}; border:1px solid {DARK_LINE}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:12px">
    <h3 style="font-family:{HEAD}; font-size:44px; font-weight:600; line-height:1.2; color:{ACCENT_D}">1.0 is the line that matters</h3>
    <p style="font-size:30px; line-height:1.5; color:{ON_DARK}">Below it, transcription finishes faster than speech arrives. Above it, every minute of lecture takes more than a minute to caption &mdash; so the captions fall behind and never catch up.</p>
  </div>
  {footer('Measured on CPU, Whisper Small, real recorded speech &#183; docs/CONCURRENCY.md', ON_DARK_DIM)}
  <aside>This is the one slide that has to land. Say it plainly: on a CPU, turning the glossary on makes the captions stop keeping up. If asked "why not a smaller model?" - because the glossary IS the feature, and a smaller model writes worse explanations. The NPU is how you keep both.</aside>
</section>
""")

# --------------------------------------------------------- 6. architecture
slide("architecture", f"""
<section id="architecture" style="{deck_light}; gap:40px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">How it fits together</h2>
  <div style="display:flex; gap:20px; align-items:stretch">
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:16px; padding:32px; display:flex; flex-direction:column; gap:12px">
      <p style="font-size:24px; color:{ON_LIGHT_DIM}; letter-spacing:1px">CPU</p>
      <h3 style="font-family:{HEAD}; font-size:32px; font-weight:600; line-height:1.2">Loopback capture</h3>
      <p style="font-size:26px; line-height:1.45; color:{ON_LIGHT_DIM}">Whatever is playing. No virtual cable.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:16px; padding:32px; display:flex; flex-direction:column; gap:12px">
      <p style="font-size:24px; color:{ON_LIGHT_DIM}; letter-spacing:1px">CPU</p>
      <h3 style="font-family:{HEAD}; font-size:32px; font-weight:600; line-height:1.2">Silero VAD</h3>
      <p style="font-size:26px; line-height:1.45; color:{ON_LIGHT_DIM}">Splits on pauses, so captions break where sentences do.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:2px solid {NPU_L}; border-radius:16px; padding:32px; display:flex; flex-direction:column; gap:12px">
      <p style="font-size:24px; color:{NPU_L}; letter-spacing:1px">NPU</p>
      <h3 style="font-family:{HEAD}; font-size:32px; font-weight:600; line-height:1.2">Whisper Small</h3>
      <p style="font-size:26px; line-height:1.45; color:{ON_LIGHT_DIM}">Transcription and language detection.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:16px; padding:32px; display:flex; flex-direction:column; gap:12px">
      <p style="font-size:24px; color:{WARN_L}; letter-spacing:1px">NPU TARGET</p>
      <h3 style="font-family:{HEAD}; font-size:32px; font-weight:600; line-height:1.2">NLLB-200</h3>
      <p style="font-size:26px; line-height:1.45; color:{ON_LIGHT_DIM}">Eight Indian languages, technical terms protected.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:16px; padding:32px; display:flex; flex-direction:column; gap:12px">
      <p style="font-size:24px; color:{WARN_L}; letter-spacing:1px">NPU TARGET</p>
      <h3 style="font-family:{HEAD}; font-size:32px; font-weight:600; line-height:1.2">Llama 3.2</h3>
      <p style="font-size:26px; line-height:1.45; color:{ON_LIGHT_DIM}">Glossary live; notes and a self-test at the end.</p>
    </div>
  </div>
  <div style="background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:12px">
    <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2">Three threads, and both queues drop the <i>oldest</i> item under pressure</h3>
    <p style="font-size:30px; line-height:1.5; color:{ON_LIGHT_DIM}">Because a caption four minutes late is worse than no caption at all. Staying near the live edge is the whole job.</p>
  </div>
  {footer('Architecture &#183; docs/ARCHITECTURE.md', ON_LIGHT_DIM)}
  <aside>Do not narrate all five boxes. Point at them once, then spend the time on the queue policy - it is the design decision that shows you thought about a live captioner rather than a batch transcriber.</aside>
</section>
""")

# ---------------------------------------------------------------- 7. proof
slide("proof", f"""
<section id="proof" style="{deck_light}; gap:36px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">Proof it is on the NPU</h2>
  <p style="font-size:32px; line-height:1.5; color:{ON_LIGHT_DIM}; width:1600px">Sahaay was built without a Snapdragon PC on the desk. Rather than claim NPU performance that could not be verified, the graphs went to Qualcomm&rsquo;s own device farm.</p>
  <table style="font-family:{BODY}; font-size:32px; color:{ON_LIGHT}">
    <tr>
      <th style="width:38%; text-align:left; font-size:24px; color:{ON_LIGHT_DIM}; font-weight:600; padding:16px">WHISPER ENCODER</th>
      <th style="width:20%; text-align:right; font-size:24px; color:{ON_LIGHT_DIM}; font-weight:600">ON-DEVICE</th>
      <th style="width:20%; text-align:right; font-size:24px; color:{ON_LIGHT_DIM}; font-weight:600">LAYERS ON NPU</th>
      <th style="width:22%; text-align:right; font-size:24px; color:{ON_LIGHT_DIM}; font-weight:600">PUBLIC JOB</th>
    </tr>
    <tr>
      <td style="text-align:left">Snapdragon X2 Elite CRD</td>
      <td style="text-align:right; color:{NPU_L}">13.5 ms</td>
      <td style="text-align:right; color:{NPU_L}">129 / 129</td>
      <td style="text-align:right; color:{ON_LIGHT_DIM}">jglyo70e5</td>
    </tr>
    <tr>
      <td style="text-align:left">Snapdragon X Elite CRD</td>
      <td style="text-align:right">27.6 ms</td>
      <td style="text-align:right; color:{NPU_L}">129 / 129</td>
      <td style="text-align:right; color:{ON_LIGHT_DIM}">j5m0o4zyg</td>
    </tr>
    <tr>
      <td style="text-align:left">Snapdragon X Plus 8-Core CRD</td>
      <td style="text-align:right">26.8 ms</td>
      <td style="text-align:right; color:{NPU_L}">129 / 129</td>
      <td style="text-align:right; color:{ON_LIGHT_DIM}">jpxl3m7jp</td>
    </tr>
  </table>
  <div style="display:flex; gap:32px">
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:20px; padding:36px; display:flex; flex-direction:column; gap:12px">
      <h3 style="font-family:{HEAD}; font-size:36px; font-weight:600; line-height:1.2; color:{NPU_L}">129 of 129 layers</h3>
      <p style="font-size:28px; line-height:1.45; color:{ON_LIGHT_DIM}">Not &ldquo;targeted the NPU&rdquo;. Qualcomm&rsquo;s own profiler reports zero CPU fallback for the whole graph.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:20px; padding:36px; display:flex; flex-direction:column; gap:12px">
      <h3 style="font-family:{HEAD}; font-size:36px; font-weight:600; line-height:1.2; color:{ACCENT_L}">The tool says &ldquo;no&rdquo;</h3>
      <p style="font-size:28px; line-height:1.45; color:{ON_LIGHT_DIM}">On x86, <b>--device</b> prints <b>Hexagon NPU active: no</b>. A tool that only ever reports success is not evidence.</p>
    </div>
  </div>
  {footer('Qualcomm AI Hub &#183; every job link is public &#183; docs/AIHUB.md', ON_LIGHT_DIM)}
  <aside>Have docs/AIHUB.md open in a tab with the URLs visible. Offer to open one. The credibility beat is the second card: show --device reporting "no" on the laptop you are presenting from.</aside>
</section>
""")

# -------------------------------------------------------------- 8. honesty
slide("honesty", f"""
<section id="honesty" style="{deck_dark}; gap:36px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">Two things I got wrong</h2>
  <div style="background:{DARK_SURFACE}; border:1px solid {DARK_LINE}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:14px">
    <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2; color:{WARN_L}">The benchmark measured the wrong model</h3>
    <p style="font-size:30px; line-height:1.5; color:{ON_DARK}">The published figure was RTF 0.042, and the product runs at 0.43. The model id is <b>auto</b>, so what got benchmarked was whichever weights happened to be on the machine &mdash; Whisper Tiny, not the Small the product ships. The harness recorded neither the model nor the signal, so nobody could check it.</p>
    <p style="font-size:30px; line-height:1.5; color:{ON_DARK_DIM}">Both harnesses now print the model and the signal beside every number. Correcting it reversed the conclusion on the previous slide &mdash; for the worse, and in the NPU&rsquo;s favour.</p>
  </div>
  <div style="background:{DARK_SURFACE}; border:1px solid {DARK_LINE}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:14px">
    <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2; color:{WARN_L}">Term protection was wired up but never seeded</h3>
    <p style="font-size:30px; line-height:1.5; color:{ON_DARK}">Telugu translated &ldquo;eigenvalues&rdquo; into &ldquo;self values&rdquo; &mdash; the exact failure the feature exists to prevent. Only real weights against speech with known ground truth caught it.</p>
    <p style="font-size:30px; line-height:1.5; color:{ON_DARK_DIM}">A feature that exists in the code is not a feature that works.</p>
  </div>
  {footer('Engineering honesty', ON_DARK_DIM)}
  <aside>This slide is worth more than another feature. Every judge on a Qualcomm panel has shipped a wrong benchmark. Mention the QNN trap too if there is time: pip install onnxruntime-qnn is not sufficient - without an explicit register_execution_provider_library call the app silently runs on CPU on the Snapdragon device itself.</aside>
</section>
""")

# ----------------------------------------------------------- 9. deployment
slide("deployment", f"""
<section id="deployment" style="{deck_light}; gap:36px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">Deployment, and what it refuses to do</h2>
  <div style="display:flex; gap:32px">
    <div style="flex:1; display:flex; flex-direction:column; gap:20px">
      <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2">Two commands</h3>
      <p style="font-size:30px; line-height:1.5; color:{ON_LIGHT_DIM}">install.ps1, then run.bat. No Node, no build step, no Docker. Every stage degrades rather than failing, and <b>--mock</b> runs the whole pipeline with nothing downloaded.</p>
      <p style="font-size:30px; line-height:1.5; color:{ON_LIGHT_DIM}"><b>259 tests</b>, green on Windows x86, Windows ARM64 and Linux &mdash; with no weights and no audio device.</p>
      <p style="font-size:30px; line-height:1.5; color:{ON_LIGHT_DIM}">Resizable captions, live regions, reduced motion, full keyboard control. Binds to 127.0.0.1. No telemetry, no account, audio never written to disk.</p>
    </div>
    <div style="flex:1; background:{LIGHT_SURFACE}; border:2px solid {ACCENT_L}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:16px">
      <h3 style="font-family:{HEAD}; font-size:40px; font-weight:600; line-height:1.2; color:{ACCENT_L}">The app is deliberately not hosted</h3>
      <p style="font-size:30px; line-height:1.5; color:{ON_LIGHT}">There is no NPU in a datacenter and no audio to capture. Shipping a student&rsquo;s lecture to a server to prove it never leaves their machine would be an odd way to make the point.</p>
      <p style="font-size:30px; line-height:1.5; color:{ON_LIGHT_DIM}">What is online is the evidence and a replay of a real session, in the real interface.</p>
      <p style="font-size:30px; line-height:1.5; color:{ACCENT_L}">sahaay-offline.vercel.app</p>
    </div>
  </div>
  {footer('Deployment and accessibility &#183; docs/WEB.md', ON_LIGHT_DIM)}
  <aside>The right-hand card is the one to dwell on. Refusing to host it is a design decision, not a gap, and saying so out loud pre-empts the obvious question.</aside>
</section>
""")

# ---------------------------------------------------------------- 10. next
slide("next", f"""
<section id="next" style="{deck_light}; gap:40px">
  <h2 style="font-family:{HEAD}; font-size:72px; font-weight:600; line-height:1.15">What is not proven yet</h2>
  <div style="display:flex; flex-direction:column; gap:24px">
    <div style="background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:16px; padding:36px; display:flex; flex-direction:column; gap:10px">
      <h3 style="font-family:{HEAD}; font-size:36px; font-weight:600; line-height:1.2">The full pipeline has never run on physical Snapdragon hardware</h3>
      <p style="font-size:28px; line-height:1.45; color:{ON_LIGHT_DIM}">Individual graphs have, on Qualcomm&rsquo;s device farm, at 129/129 layers. The suite runs on ARM64 Windows in CI &mdash; same instruction set, Microsoft silicon, no Hexagon. That proves the install path, not the NPU.</p>
    </div>
    <div style="background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:16px; padding:36px; display:flex; flex-direction:column; gap:10px">
      <h3 style="font-family:{HEAD}; font-size:36px; font-weight:600; line-height:1.2">Word error rate is measured against synthesised speech</h3>
      <p style="font-size:28px; line-height:1.45; color:{ON_LIGHT_DIM}">No accent, no room, no crosstalk, no disfluency. The published figure is a floor, and the report says so. Real speakers are next.</p>
    </div>
    <div style="background:{LIGHT_SURFACE}; border:1px solid {LIGHT_LINE}; border-radius:16px; padding:36px; display:flex; flex-direction:column; gap:10px">
      <h3 style="font-family:{HEAD}; font-size:36px; font-weight:600; line-height:1.2">The NPU column of the concurrency table is empty</h3>
      <p style="font-size:28px; line-height:1.45; color:{ON_LIGHT_DIM}">Slide five is the CPU half of the comparison. The half that closes the argument needs a Snapdragon PC and one command.</p>
    </div>
  </div>
  {footer('Stated here rather than left for a judge to find', ON_LIGHT_DIM)}
  <aside>Say these before anyone asks. Volunteering the gaps is what makes the measured claims believable - and it turns the obvious hostile question into a conversation you started.</aside>
</section>
""")

# --------------------------------------------------------------- 11. close
slide("close", f"""
<section id="close" data-transition="fade" style="{deck_dark}; justify-content:center; gap:40px">
  <p style="font-size:24px; color:{ACCENT_D}; letter-spacing:2px; text-transform:uppercase">If you remember one sentence</p>
  <h1 style="font-family:{HEAD}; font-size:76px; font-weight:600; line-height:1.25; width:1600px">On a CPU, turning on the glossary makes the captions stop keeping up with the lecturer. That is why this is a Snapdragon application.</h1>
  <p style="font-size:32px; line-height:1.5; color:{ON_DARK_DIM}; width:1500px">Everything else &mdash; the accuracy table, the term protection, the degradation ladder &mdash; supports that sentence.</p>
  {footer('sahaay-offline.vercel.app  &#183;  github.com/andringodson/sahaay  &#183;  MIT', ON_DARK_DIM)}
  <aside>Land the sentence, then stop talking. Let the silence do the closing.</aside>
</section>
""")

# ------------------------------------------------------------------ index
order = [
    "cover", "problem", "who", "demo", "npu",
    "architecture", "proof", "honesty", "deployment", "next", "close",
]

index = {
    "v": 4,
    "createdOnFiles": {"v": 1, "at": "2026-09-22T15:00:00Z"},
    "title": "Sahaay - Snapdragon AI Lab 2026",
    "order": order,
    "sections": {
        "why": {"description": "Who needs this and why cloud captioning does not solve it", "start": "cover"},
        "argument": {"description": "The measurement that makes it a Snapdragon application", "start": "demo"},
        "evidence": {"description": "Proof it runs on the NPU, and what is still unproven", "start": "proof"},
    },
    "faces": {
        "rubik": {
            "family": "Rubik",
            "href": "https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap",
        },
        "ibm-plex-sans": {
            "family": "IBM Plex Sans",
            "href": "https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap",
        },
    },
    "designSystems": [],
}

for sid, html in SLIDE_FILES.items():
    (SLIDES / f"{sid}.html").write_text(html, encoding="utf-8")

(ROOT / "project" / "deck.json").write_text(
    json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
)

missing = [s for s in order if s not in SLIDE_FILES]
extra = [s for s in SLIDE_FILES if s not in order]
print(f"wrote {len(SLIDE_FILES)} slides + deck.json at {ROOT}")
print(f"  order    {order}")
print(f"  missing  {missing or 'none'}")
print(f"  extra    {extra or 'none'}")
