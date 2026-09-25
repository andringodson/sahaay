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

ON THE LOOK. One effect idea, used twice: a radial glow on the two bookend
slides and flat toned backgrounds everywhere else. The chart colours are not
a taste decision - they were run through the dataviz validator, because the
obvious green/red pair fails on deuteranopia (delta-E 8.3, a hair over the
floor) and also sits outside the dark-mode lightness band. The pair used here
clears every check at delta-E 23.4, and the bars carry their values as text
so nothing depends on colour alone.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
SLIDES = ROOT / "project" / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)

# The product's own palette, deepened for projection.
INK = "#0A0E14"
CARD_D = "#131B26"
LINE_D = "rgba(255,255,255,0.12)"
ON_D = "#E9EFF6"
ON_D_DIM = "#93A3B5"

PAPER = "#F7F9FC"
CARD_L = "#FFFFFF"
LINE_L = "#DCE3EC"
ON_L = "#0F1720"
ON_L_DIM = "#56646F"

ACCENT_D = "#5CBFFF"     # bright enough to read as text on ink
ACCENT_L = "#0B6FB8"     # 4.5:1 on paper
NPU_L = "#00734F"
WARN = "#FFB020"
WARN_L = "#8A5800"

# Validated with dataviz/scripts/validate_palette.js --mode dark:
# lightness band, chroma floor, CVD separation (dE 23.4 protan), normal-vision
# floor and contrast all pass. Do not swap these for green/red.
BAR_OK = "#2D8FD6"
BAR_BAD = "#DE4E36"

GLOW_TL = f"radial-gradient(ellipse at 22% 12%, #1B4E80 0%, {INK} 62%)"
GLOW_BR = f"radial-gradient(ellipse at 78% 88%, #1B4E80 0%, {INK} 62%)"

HEAD = "'Rubik', Verdana, sans-serif"
BODY = "'IBM Plex Sans', Verdana, sans-serif"

SHOT = "/_blob/3a9e655f5a0b3cee28c0f67817bc8d83"   # docs/img/demo.png

dark = (
    f"background:{INK}; color:{ON_D}; font-family:{BODY}; "
    "display:flex; flex-direction:column; padding:128px 128px 160px"
)
light = (
    f"background:{PAPER}; color:{ON_L}; font-family:{BODY}; "
    "display:flex; flex-direction:column; padding:128px 128px 160px"
)


def footer(text, color):
    return (
        f'<p style="position:absolute; left:128px; bottom:64px; width:1664px; '
        f'font-size:24px; color:{color}">{text}</p>'
    )


def eyebrow(text, color):
    return (
        f'<p style="font-size:24px; color:{color}; letter-spacing:3px; '
        f'text-transform:uppercase; font-weight:600">{text}</p>'
    )


def card_d(inner, pad=40, border=LINE_D):
    return (
        f'<div style="background:{CARD_D}; border:1px solid {border}; '
        f'border-radius:20px; padding:{pad}px; display:flex; '
        f'flex-direction:column; gap:14px">{inner}</div>'
    )


def card_l(inner, pad=40, border=LINE_L, flex=False):
    grow = "flex:1; " if flex else ""
    return (
        f'<div style="{grow}background:{CARD_L}; border:1px solid {border}; '
        f'border-radius:20px; padding:{pad}px; display:flex; '
        f'flex-direction:column; gap:14px; '
        f'box-shadow:0 4px 24px rgba(15,23,32,0.06)">{inner}</div>'
    )


def waveform():
    """An audio waveform settling into caption lines.

    The one piece of drawn art in the deck, on the cover only. Heights are
    generated rather than hand-placed so it reads as a signal, not a pattern.
    No <text> inside: fonts never load in an SVG.
    """
    import math

    bars = []
    count = 96
    for i in range(count):
        # Two beating sines plus a decaying envelope: loud at the left,
        # settling toward the right, the way a sentence does.
        t = i / (count - 1)
        amp = (
            0.55 * math.sin(i * 0.55)
            + 0.30 * math.sin(i * 0.23 + 1.1)
            + 0.15 * math.sin(i * 1.30 + 0.4)
        )
        env = 0.35 + 0.65 * (1.0 - t) ** 1.4
        h = max(6.0, abs(amp) * 108.0 * env + 6.0)
        x = i * 17.2
        y = (120.0 - h) / 2.0
        opacity = round(0.20 + 0.65 * env, 3)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="8" height="{h:.1f}" '
            f'rx="4" fill="#5CBFFF" opacity="{opacity}"/>'
        )
    return (
        '<svg width="1664" height="120" viewBox="0 0 1664 120" '
        'aria-label="An audio waveform settling from loud speech into quiet">'
        + "".join(bars)
        + "</svg>"
    )


SLIDE_FILES = {}


def slide(sid, html):
    SLIDE_FILES[sid] = html.strip() + "\n"


# ---------------------------------------------------------------- 1. cover
slide("cover", f"""
<section id="cover" data-transition="fade" style="background:{GLOW_TL}; color:{ON_D}; font-family:{BODY}; display:flex; flex-direction:column; justify-content:center; padding:128px 128px 160px; gap:36px">
  {eyebrow('Snapdragon&#174; AI Lab Build &amp; Present Challenge 2026', ACCENT_D)}
  <h1 style="font-family:{HEAD}; font-size:170px; font-weight:600; line-height:0.98; letter-spacing:-4px">Sahaay</h1>
  <h2 style="font-family:{HEAD}; font-size:58px; font-weight:400; line-height:1.15; width:1400px">The lecture understands you &mdash; offline</h2>
  <p style="font-size:32px; color:{ON_D_DIM}; width:1280px; line-height:1.45">Live captions, Indian-language translation and a jargon glossary, running entirely on a Snapdragon-powered PC with the network switched off.</p>
  <div style="opacity:0.9">{waveform()}</div>
  {footer('sahaay-offline.vercel.app  &#183;  github.com/andringodson/sahaay', ON_D_DIM)}
  <aside>Open with the name and one sentence, not the tech stack. Leave the URL on screen: a judge who wants to check a number can, mid-talk.</aside>
</section>
""")

# -------------------------------------------------------------- 2. problem
slide("problem", f"""
<section id="problem" style="{light}; gap:44px">
  {eyebrow('The problem', ACCENT_L)}
  <h2 style="font-family:{HEAD}; font-size:68px; font-weight:600; line-height:1.12; letter-spacing:-1px">An engineering lecture in India is rarely in one language</h2>
  <div style="background:{CARD_L}; border-left:8px solid {ACCENT_L}; border-radius:8px; padding:48px 56px; box-shadow:0 4px 24px rgba(15,23,32,0.06)">
    <p style="font-size:46px; line-height:1.35; color:{ON_L}">&ldquo;Matrix A ka <b style="color:{ACCENT_L}">determinant</b> zero hoga, tabhi non-trivial <b style="color:{ACCENT_L}">solution</b> milega.&rdquo;</p>
  </div>
  <div style="display:flex; gap:32px">
    <p style="flex:1; font-size:30px; line-height:1.45; color:{ON_L_DIM}">A student following that in their second language has two problems at once: parse a Hindi sentence, <i>and</i> hold on to the English technical terms &mdash; the exact terms that appear in the textbook and the exam.</p>
    <p style="flex:1; font-size:30px; line-height:1.45; color:{ON_L_DIM}">Cloud captioning handles code-mixing badly, costs money per minute, and needs connectivity a lecture hall, a hostel or a rural college often does not have.</p>
  </div>
  {footer('The problem', ON_L_DIM)}
  <aside>Read the sentence aloud. It does the work: everyone in the room recognises it instantly.</aside>
</section>
""")

# ------------------------------------------------------------------ 3. who
_who = []
for icon, title, body in [
    ("Chat", "Deaf and hard of hearing", "A cloud round-trip is too slow to follow a sentence in progress."),
    ("Globe", "Studying in a second language", "The lecture is in English; the thinking is in Hindi, Tamil, Bengali."),
    ("Cloud", "No usable connectivity", "A hall, a hostel, a bus. Bandwidth is not a given."),
]:
    _who.append(f"""
    <div style="flex:1; background:{CARD_L}; border:1px solid {LINE_L}; border-radius:20px; padding:44px; display:flex; flex-direction:column; gap:20px; box-shadow:0 4px 24px rgba(15,23,32,0.06)">
      <div style="width:84px; height:84px; border-radius:50%; background:#E3F0FA; display:flex; align-items:center; justify-content:center">
        <x-icon name="{icon}" style="width:44px; height:44px; color:{ACCENT_L}"></x-icon>
      </div>
      <h3 style="font-family:{HEAD}; font-size:38px; font-weight:600; line-height:1.18">{title}</h3>
      <p style="font-size:28px; line-height:1.45; color:{ON_L_DIM}">{body}</p>
    </div>""")

slide("who", f"""
<section id="who" style="{light}; gap:48px">
  {eyebrow('Who this is for', ACCENT_L)}
  <h2 style="font-family:{HEAD}; font-size:68px; font-weight:600; line-height:1.12; letter-spacing:-1px">Three people, one product</h2>
  <div style="display:flex; gap:32px">{"".join(_who)}</div>
  <p style="font-size:34px; line-height:1.45; color:{ON_L}; width:1600px">Not a hypothetical persona. This is the majority experience in Indian engineering education.</p>
  {footer('Who this is for', ON_L_DIM)}
  <aside>Do not linger. One breath per card, then the last line slowly.</aside>
</section>
""")

# ----------------------------------------------------------------- 4. demo
slide("demo", f"""
<section id="demo" data-transition="push" style="{dark}; gap:44px">
  <div style="display:flex; gap:56px; align-items:center">
    <div style="width:600px; display:flex; flex-direction:column; gap:28px">
      {eyebrow('Demo', ACCENT_D)}
      <h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; letter-spacing:-1px">A lecture plays. The captions keep up.</h2>
      <p style="font-size:30px; line-height:1.45; color:{ON_D_DIM}">Hindi captions with the English technical terms left intact, and the jargon explaining itself in the sidebar as it is spoken.</p>
      <p style="font-size:34px; line-height:1.4; color:{ACCENT_D}">Then I turn the Wi-Fi off, and nothing changes.</p>
    </div>
    <img src="{SHOT}" alt="The Sahaay interface: English captions with Hindi translations beneath each line, and a jargon sidebar explaining eigenvector and eigenvalue" style="width:1008px; height:630px; object-fit:contain; border:1px solid {LINE_D}; border-radius:16px; box-shadow:0 12px 48px rgba(0,0,0,0.45)">
  </div>
  {footer('Live demo &#183; fallback replay at sahaay-offline.vercel.app/demo', ON_D_DIM)}
  <aside>Turn the Wi-Fi off on camera. It is the strongest fifteen seconds you have - do not cut it, and do not explain it first. Two fallbacks if the machine misbehaves: sahaay-offline.vercel.app/live runs Whisper in the browser on any laptop in the room, and /demo replays a recorded session in the same interface.</aside>
</section>
""")

# ------------------------------------------------------------------ 5. npu
# 0 to 2.0 RTF across 1568px of card interior: 784px per 1.0.
PER_RTF = 784.0


def bar(label, rtf, ms, colour):
    width = round(rtf * PER_RTF)
    return f"""
    <div style="display:flex; flex-direction:column; gap:14px">
      <div style="display:flex; justify-content:space-between; align-items:baseline">
        <p style="font-size:30px; color:{ON_D}">{label}</p>
        <p style="font-size:34px; color:{colour}; font-weight:600">RTF {rtf} &#183; {ms}</p>
      </div>
      <div style="height:56px; background:#1B2532; border-radius:4px">
        <div style="width:{width}px; height:56px; background:{colour}; border-radius:0 4px 4px 0"></div>
      </div>
    </div>"""


slide("npu", f"""
<section id="npu" style="{dark}; gap:32px">
  {eyebrow('Why this needs an NPU', ACCENT_D)}
  <h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; letter-spacing:-1px">Turn the glossary on, and a CPU stops keeping up</h2>
  <div style="background:{CARD_D}; border:1px solid {LINE_D}; border-radius:20px; padding:48px">
    <div style="position:relative; display:flex; flex-direction:column; gap:32px">
      <div style="position:absolute; left:783px; top:56px; width:3px; height:284px; background:{WARN}"></div>
      <p style="position:absolute; left:659px; top:0; width:250px; font-size:26px; color:{WARN}; font-weight:600; text-align:center">1.0 &#183; real time</p>
      <div style="height:44px"></div>
      {bar('Glossary idle', 0.41, '3276 ms', BAR_OK)}
      {bar('Glossary running', 1.55, '12400 ms', BAR_BAD)}
    </div>
  </div>
  <p style="font-size:32px; line-height:1.45; color:{ON_D}; width:1600px">Below the line, transcription finishes faster than speech arrives. Above it, every minute of lecture takes more than a minute to caption &mdash; so the captions fall behind and never catch up.</p>
  {footer('CPU, Whisper Small, real recorded speech &#183; docs/CONCURRENCY.md', ON_D_DIM)}
  <aside>This is the one slide that has to land. Point at the orange line and say it plainly: on a CPU, turning the glossary on pushes the captions past real time. If asked "why not a smaller model?" - because the glossary IS the feature, and a smaller model writes worse explanations. The NPU is how you keep both.</aside>
</section>
""")

# --------------------------------------------------------- 6. architecture
_stages = []
for tag, tag_col, chip, name, body in [
    ("CPU", ON_L_DIM, "#EDF1F6", "Loopback capture", "Whatever is playing. No virtual cable, no driver."),
    ("CPU", ON_L_DIM, "#EDF1F6", "Silero VAD", "Splits on pauses, so captions break where sentences do."),
    ("NPU", NPU_L, "#DFF3EA", "Whisper Small", "Transcription and per-segment language detection."),
    ("NPU TARGET", WARN_L, "#FBF0DC", "NLLB-200", "Eight Indian languages, technical terms protected."),
    ("NPU TARGET", WARN_L, "#FBF0DC", "Llama 3.2", "Glossary live; notes and a self-test at the end."),
]:
    _stages.append(f"""
    <div style="flex:1; background:{CARD_L}; border:1px solid {LINE_L}; border-radius:18px; padding:30px; display:flex; flex-direction:column; gap:14px; box-shadow:0 4px 24px rgba(15,23,32,0.06)">
      <p style="font-size:24px; color:{tag_col}; background:{chip}; border-radius:8px; padding:6px 12px; letter-spacing:1px; font-weight:600">{tag}</p>
      <h3 style="font-family:{HEAD}; font-size:32px; font-weight:600; line-height:1.15">{name}</h3>
      <p style="font-size:25px; line-height:1.4; color:{ON_L_DIM}">{body}</p>
    </div>""")

slide("architecture", f"""
<section id="architecture" style="{light}; gap:36px">
  {eyebrow('Architecture', ACCENT_L)}
  <h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; letter-spacing:-1px">How it fits together</h2>
  <div style="display:flex; gap:18px; align-items:stretch">{"".join(_stages)}</div>
  <div style="background:{CARD_L}; border-left:8px solid {ACCENT_L}; border-radius:8px; padding:40px 48px; display:flex; flex-direction:column; gap:12px; box-shadow:0 4px 24px rgba(15,23,32,0.06)">
    <h3 style="font-family:{HEAD}; font-size:38px; font-weight:600; line-height:1.18">Three threads, and both queues drop the <i>oldest</i> item under pressure</h3>
    <p style="font-size:29px; line-height:1.45; color:{ON_L_DIM}">Because a caption four minutes late is worse than no caption at all. Staying near the live edge is the whole job.</p>
  </div>
  {footer('Architecture &#183; docs/ARCHITECTURE.md', ON_L_DIM)}
  <aside>Do not narrate all five boxes. Point at them once, then spend the time on the queue policy - it is the design decision that shows you thought about a live captioner rather than a batch transcriber.</aside>
</section>
""")

# ---------------------------------------------------------------- 7. proof
slide("proof", f"""
<section id="proof" style="{light}; gap:32px">
  {eyebrow('Proof it is on the NPU', ACCENT_L)}
  <h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; letter-spacing:-1px">Measured on Qualcomm&rsquo;s own device farm</h2>
  <table style="font-family:{BODY}; font-size:31px; color:{ON_L}">
    <tr>
      <th style="width:38%; text-align:left; font-size:24px; color:{ON_L_DIM}; font-weight:600; padding:14px">WHISPER ENCODER</th>
      <th style="width:20%; text-align:right; font-size:24px; color:{ON_L_DIM}; font-weight:600">ON-DEVICE</th>
      <th style="width:20%; text-align:right; font-size:24px; color:{ON_L_DIM}; font-weight:600">LAYERS ON NPU</th>
      <th style="width:22%; text-align:right; font-size:24px; color:{ON_L_DIM}; font-weight:600">PUBLIC JOB</th>
    </tr>
    <tr>
      <td style="text-align:left"><b>Snapdragon X2 Elite CRD</b></td>
      <td style="text-align:right; color:{NPU_L}"><b>13.5 ms</b></td>
      <td style="text-align:right; color:{NPU_L}"><b>129 / 129</b></td>
      <td style="text-align:right; color:{ON_L_DIM}">jglyo70e5</td>
    </tr>
    <tr>
      <td style="text-align:left">Snapdragon X Elite CRD</td>
      <td style="text-align:right">27.6 ms</td>
      <td style="text-align:right; color:{NPU_L}">129 / 129</td>
      <td style="text-align:right; color:{ON_L_DIM}">j5m0o4zyg</td>
    </tr>
    <tr>
      <td style="text-align:left">Snapdragon X Plus 8-Core CRD</td>
      <td style="text-align:right">26.8 ms</td>
      <td style="text-align:right; color:{NPU_L}">129 / 129</td>
      <td style="text-align:right; color:{ON_L_DIM}">jpxl3m7jp</td>
    </tr>
  </table>
  <div style="display:flex; gap:32px">
    {card_l(f'<h3 style="font-family:{HEAD}; font-size:34px; font-weight:600; line-height:1.18; color:{NPU_L}">129 of 129 layers</h3><p style="font-size:27px; line-height:1.4; color:{ON_L_DIM}">Not &ldquo;targeted the NPU&rdquo;. Qualcomm&rsquo;s own profiler reports zero CPU fallback for the whole graph.</p>', pad=34, flex=True)}
    {card_l(f'<h3 style="font-family:{HEAD}; font-size:34px; font-weight:600; line-height:1.18; color:{ACCENT_L}">The tool says &ldquo;no&rdquo;</h3><p style="font-size:27px; line-height:1.4; color:{ON_L_DIM}">On x86, <b>--device</b> prints <b>Hexagon NPU active: no</b>. A tool that only ever reports success is not evidence.</p>', pad=34, flex=True)}
  </div>
  {footer('Every job link is public &#183; docs/AIHUB.md', ON_L_DIM)}
  <aside>Have docs/AIHUB.md open in a tab with the URLs visible. Offer to open one. The credibility beat is the second card: show --device reporting "no" on the laptop you are presenting from.</aside>
</section>
""")

# -------------------------------------------------------------- 8. honesty
slide("honesty", f"""
<section id="honesty" style="{dark}; gap:32px">
  {eyebrow('Engineering honesty', WARN)}
  <h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; letter-spacing:-1px">Two things I got wrong</h2>
  {card_d(f'<h3 style="font-family:{HEAD}; font-size:38px; font-weight:600; line-height:1.18; color:{WARN}">The benchmark measured the wrong model</h3><p style="font-size:29px; line-height:1.45; color:{ON_D}">The published figure was RTF 0.042; the product runs at 0.43. The model id is <b>auto</b>, so what got benchmarked was whichever weights happened to be on the machine &mdash; Whisper Tiny, not the Small the product ships. Neither the model nor the signal was recorded, so nobody could check it.</p><p style="font-size:29px; line-height:1.45; color:{ON_D_DIM}">Both harnesses now print the model and the signal beside every number. Correcting it reversed the previous slide &mdash; for the worse, and in the NPU&rsquo;s favour.</p>')}
  {card_d(f'<h3 style="font-family:{HEAD}; font-size:38px; font-weight:600; line-height:1.18; color:{WARN}">Term protection was wired up but never seeded</h3><p style="font-size:29px; line-height:1.45; color:{ON_D}">Telugu translated &ldquo;eigenvalues&rdquo; into &ldquo;self values&rdquo; &mdash; the exact failure the feature exists to prevent. Only real weights against speech with known ground truth caught it.</p><p style="font-size:29px; line-height:1.45; color:{ON_D_DIM}">A feature that exists in the code is not a feature that works.</p>')}
  {footer('Engineering honesty', ON_D_DIM)}
  <aside>This slide is worth more than another feature. Every judge on a Qualcomm panel has shipped a wrong benchmark. Mention the QNN trap too if there is time: pip install onnxruntime-qnn is not sufficient - without an explicit register_execution_provider_library call the app silently runs on CPU on the Snapdragon device itself.</aside>
</section>
""")

# ----------------------------------------------------------- 9. deployment
slide("deployment", f"""
<section id="deployment" style="{light}; gap:36px">
  {eyebrow('Deployment and accessibility', ACCENT_L)}
  <h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; letter-spacing:-1px">Two commands &mdash; and one deliberate refusal</h2>
  <div style="display:flex; gap:36px; align-items:stretch">
    <div style="flex:1; display:flex; flex-direction:column; gap:24px">
      <p style="font-size:29px; line-height:1.45; color:{ON_L_DIM}"><b style="color:{ON_L}">install.ps1</b>, then <b style="color:{ON_L}">run.bat</b>. No Node, no build step, no Docker. Every stage degrades rather than failing, and <b style="color:{ON_L}">--mock</b> runs the whole pipeline with nothing downloaded.</p>
      <p style="font-size:29px; line-height:1.45; color:{ON_L_DIM}"><b style="color:{ON_L}">259 tests</b>, green on Windows x86, Windows ARM64 and Linux &mdash; with no weights and no audio device.</p>
      <p style="font-size:29px; line-height:1.45; color:{ON_L_DIM}">Resizable captions, live regions, reduced motion, full keyboard control. Binds to 127.0.0.1. No telemetry, no account, audio never written to disk.</p>
    </div>
    <div style="flex:1; background:{CARD_L}; border:2px solid {ACCENT_L}; border-radius:20px; padding:40px; display:flex; flex-direction:column; gap:16px; box-shadow:0 4px 24px rgba(15,23,32,0.06)">
      <h3 style="font-family:{HEAD}; font-size:38px; font-weight:600; line-height:1.18; color:{ACCENT_L}">Nothing is hosted. Try it anyway.</h3>
      <p style="font-size:29px; line-height:1.45; color:{ON_L}">The site runs Whisper in <i>your</i> browser &mdash; static files, your own CPU, audio that never leaves the tab. The same claim the desktop app makes, on hardware you already have.</p>
      <p style="font-size:29px; line-height:1.45; color:{ON_L_DIM}">Watch the RTF badge climb past 1.0 while it runs. That is this deck&rsquo;s central measurement, happening to you.</p>
      <p style="font-size:30px; line-height:1.4; color:{ACCENT_L}">sahaay-offline.vercel.app/live</p>
    </div>
  </div>
  {footer('Deployment and accessibility &#183; docs/WEB.md', ON_L_DIM)}
  <aside>The right-hand card is the best invitation in the deck: tell them to open it on their phone now. A judge who watches RTF cross 1.0 on their own device has understood slide five without you saying a word. The pipeline still is not hosted - no NPU, no system audio, no translation - and saying that out loud pre-empts the obvious question.</aside>
</section>
""")

# ---------------------------------------------------------------- 10. next
_gaps = []
for title, body in [
    ("The full pipeline has never run on physical Snapdragon hardware",
     "Individual graphs have, on Qualcomm&rsquo;s device farm, at 129/129 layers. The suite runs on ARM64 Windows in CI &mdash; same instruction set, Microsoft silicon, no Hexagon. That proves the install path, not the NPU."),
    ("Word error rate is measured against synthesised speech",
     "No accent, no room, no crosstalk, no disfluency. The published figure is a floor, and the report says so. Real speakers are next."),
    ("The NPU column of the concurrency table is empty",
     "Slide five is the CPU half of the comparison. The half that closes the argument needs a Snapdragon PC and one command."),
]:
    _gaps.append(f"""
    <div style="background:{CARD_L}; border:1px solid {LINE_L}; border-left:8px solid {WARN_L}; border-radius:8px; padding:32px 40px; display:flex; flex-direction:column; gap:10px">
      <h3 style="font-family:{HEAD}; font-size:34px; font-weight:600; line-height:1.18">{title}</h3>
      <p style="font-size:27px; line-height:1.4; color:{ON_L_DIM}">{body}</p>
    </div>""")

slide("next", f"""
<section id="next" style="{light}; gap:36px">
  {eyebrow('What is next', WARN_L)}
  <h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; letter-spacing:-1px">What is not proven yet</h2>
  <div style="display:flex; flex-direction:column; gap:22px">{"".join(_gaps)}</div>
  {footer('Stated here rather than left for a judge to find', ON_L_DIM)}
  <aside>Say these before anyone asks. Volunteering the gaps is what makes the measured claims believable - and it turns the obvious hostile question into a conversation you started.</aside>
</section>
""")

# --------------------------------------------------------------- 11. close
slide("close", f"""
<section id="close" data-transition="fade" style="background:{GLOW_BR}; color:{ON_D}; font-family:{BODY}; display:flex; flex-direction:column; justify-content:center; padding:128px 128px 160px; gap:40px">
  {eyebrow('If you remember one sentence', ACCENT_D)}
  <h1 style="font-family:{HEAD}; font-size:78px; font-weight:600; line-height:1.22; letter-spacing:-1px; width:1600px">On a CPU, turning on the glossary makes the captions stop keeping up with the lecturer. That is why this is a <span style="color:{ACCENT_D}">Snapdragon</span> application.</h1>
  <p style="font-size:32px; line-height:1.45; color:{ON_D_DIM}; width:1500px">Everything else &mdash; the accuracy table, the term protection, the degradation ladder &mdash; supports that sentence.</p>
  {footer('sahaay-offline.vercel.app  &#183;  github.com/andringodson/sahaay  &#183;  MIT', ON_D_DIM)}
  <aside>Land the sentence, then stop talking. Let the silence do the closing.</aside>
</section>
""")

# ------------------------------------------------------------------ index
order = [
    "cover", "problem", "who", "demo", "npu",
    "architecture", "proof", "honesty", "deployment", "next", "close",
]

FONT_HREF = (
    "https://fonts.googleapis.com/css2?family=Rubik:wght@400;600"
    "&family=IBM+Plex+Sans:wght@400;600&display=swap"
)

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
        "rubik": {"family": "Rubik", "href": FONT_HREF},
        "ibm-plex-sans": {"family": "IBM Plex Sans", "href": FONT_HREF},
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
biggest = max(SLIDE_FILES.items(), key=lambda kv: len(kv[1]))
print(f"wrote {len(SLIDE_FILES)} slides + deck.json at {ROOT}")
print(f"  missing  {missing or 'none'}")
print(f"  extra    {extra or 'none'}")
print(f"  largest  {biggest[0]}.html at {len(biggest[1]) / 1024:.1f} KB")
