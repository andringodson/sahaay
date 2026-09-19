# Presentation kit

Everything needed for the "Present" half of the challenge: a slide outline, a
three-minute demo script, and the checklist to run before recording.

The judging criteria are **Technical Implementation**, **Application Use Case
& Innovation**, **Deployment & Accessibility**, and **Presentation &
Documentation**. Each slide below notes which one it is buying.

---

## Slides (10)

### 1 — Title
**Sahaay — the lecture understands you, offline**
Live captions, Indian-language translation and a jargon glossary, running
entirely on a Snapdragon-powered HP PC.

*Do not open with the tech stack.* Open with the name and one sentence.

### 2 — The problem *(Use Case)*
Put this sentence on the slide and read it aloud:

> "Matrix A ka **determinant** zero hoga, tabhi non-trivial **solution** milega."

Then: a student following this in their second language is parsing a Hindi
sentence *and* trying to hold onto English technical terms — the exact terms
that appear in the textbook and the exam. Cloud captioning is poor at
code-mixing, costs per minute, and needs connectivity a lecture hall often
does not have.

### 3 — Who this is for *(Use Case, Accessibility)*
- Students who are deaf or hard of hearing — a cloud round-trip is too slow mid-sentence
- Students whose first language is not the language of instruction
- Anyone in a hall, hostel or bus with no usable connectivity

Not a hypothetical persona. This is the majority experience in Indian
engineering education.

### 4 — Demo *(everything)*
Live, not a video-in-a-video if you can avoid it. See the script below.

### 5 — Why this needs an NPU *(Technical)*
The one slide that has to land.

Two models run **concurrently and continuously for a whole lecture**: Whisper
transcribing every few seconds, Llama 3.2 3B writing glossary entries
alongside it. On CPU that starves the captions or flattens the battery. On
Hexagon both graphs stay resident.

A one-hour lecture, daily, per student is also exactly the workload that is
absurd to send to the cloud — and exactly what an idle 45 TOPS NPU is for.

### 6 — Architecture *(Technical)*
Use the diagram from the README. Spend your time on one point: three threads,
and both queues drop the **oldest** item under pressure, because a caption
four minutes late is worse than no caption.

### 7 — Proof it is on the NPU *(Technical)*
This is the slide with the receipts. Put the table up and let it speak:

| Whisper encoder | On-device | Layers on NPU |
|---|---:|---:|
| Snapdragon X Elite | **27.48 ms** | **129 / 129** |
| Snapdragon X2 Elite | **13.5 ms** | **129 / 129** |

Say the two things that matter:

1. **129 out of 129 layers ran on the Hexagon NPU.** Not "targeted the NPU" —
   Qualcomm's own profiler reports zero CPU fallback for the entire graph.
2. **These are public job links.** Anyone can open them and check. Have
   `docs/AIHUB.md` on screen with the URLs visible.

Then the credibility beat: **the tool reports "no" when the answer is no.**
Show `--device` on the x86 machine printing `Hexagon NPU active : no`. A tool
that only ever reports success is not evidence of anything.

### 8 — Engineering honesty *(Technical, Documentation)*
The QNN plugin registration trap, in one slide (see `docs/HARDWARE.md`):
`pip install onnxruntime-qnn` is *not* sufficient — without an explicit
`register_execution_provider_library` call, the app silently runs on CPU on
the Snapdragon device itself.

This slide is worth more than another feature. It shows you measured rather
than assumed, and every judge on a Qualcomm panel has hit this.

If you have time for a second beat, the term-protection bug is the better
story because it is about the *product*, not the plumbing: protection was
wired up but never seeded, so Telugu translated "eigenvalues" into "self
values" — the exact failure the feature exists to prevent. It was only
caught by running real weights against speech with known ground truth.
The lesson lands: a feature that exists in the code is not a feature that
works.

### 9 — Deployment *(Deployment & Accessibility)*
- `install.ps1` → `run.bat`. No Node, no build step, no Docker
- Degrades on every stage rather than failing; `--mock` runs with nothing downloaded
- 145 tests, CI on Windows and Linux, no weights or audio device needed
- Accessibility: resizable captions, `aria-live`, reduced-motion, full keyboard control
- Privacy: binds to `127.0.0.1`, no telemetry, no accounts, audio never written to disk

### 10 — What is next
- Real WER measurement on a code-mixed Indian-English corpus
- Speaker diarisation, so "the lecturer" and "a question from the back" separate
- An Android build — the same QNN graphs already target Snapdragon mobile

---

## Three-minute demo script

**0:00 – 0:20 — the problem, out loud**
Play four seconds of a real code-mixed lecture. Say: "Half the sentence is
Hindi, the technical terms are English, and there is no internet in this
room."

**0:20 – 0:35 — start it**
`run.bat`. Point at the EP badge: *Hexagon NPU (QNN)*. Say "everything from
here is on this laptop. I am going to turn the Wi-Fi off."

**Turn the Wi-Fi off on camera.** This is the strongest fifteen seconds in
the whole video. Do not skip it.

**0:35 – 1:30 — captions and translation**
Play the lecture. Captions appear roughly a sentence behind. Switch the
language dropdown to Tamil mid-lecture and let the next caption arrive
translated.

Then point at one translated line and say: *"eigenvalue is still
eigenvalue."* Explain in one sentence that technical terms are protected
before translation, because a transliterated term cannot be matched to the
textbook.

**1:30 – 2:05 — the glossary (the hook)**
Wait for the lecturer to say a technical term and let the sidebar fill in
real time. Say: "that is a 3B language model running on the NPU, at the same
time as speech recognition, without the captions slowing down."

Have the RTF pill visible. `RTF 0.2` on screen while both models run is the
entire technical argument, made visually.

**2:05 – 2:35 — the notes**
Press Stop. The notes sheet slides up with topics, key points, glossary and a
five-question self-test. Say: "saved as Markdown, on this machine, and the
student keeps it."

**2:35 – 3:00 — the proof**
Cut to a terminal. `run.bat --device` showing *Hexagon NPU active : yes*.
Then `docs/BENCHMARKS.md` and an AI Hub job link. Close on: "measured, with
links anyone can open."

---

## Before you record

- [ ] `run.bat --device` shows **Hexagon NPU active : yes** on the demo machine
- [ ] `python scripts/bench.py --compare --write` — real NPU vs CPU numbers
- [ ] `python scripts/aihub_profile.py --all --write` — job links in `docs/AIHUB.md`
- [ ] README performance section updated with the real figures
- [ ] Lecture clip chosen: genuinely code-mixed, with at least three technical terms
- [ ] Audio routing checked — loopback captures the clip without a virtual cable
- [ ] Caption size bumped one step for video legibility
- [ ] Notes sheet tested end to end
- [ ] Wi-Fi toggle rehearsed on camera
- [ ] Repo is public, CI green, no weights committed

## Two questions you will be asked

**"Why not just use the cloud?"**
Continuity, cost and privacy, in that order. A lecture hall with 200 students
streaming audio needs bandwidth that is not there, costs per minute per
student, and sends the room's conversation to a third party. The NPU is
already in the laptop and otherwise idle.

**"How much of this is really on the NPU?"**
Whisper, NLLB and Llama, all three, via the QNN execution provider. The VAD
stays on CPU on purpose — it is 1.8 MB, and the transfer overhead would cost
more than the compute saves. `--device` and `docs/BENCHMARKS.md` are the
receipts, and `docs/AIHUB.md` cross-checks them against Qualcomm's own
device farm.
