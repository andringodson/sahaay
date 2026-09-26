# Tuning: what changed, and what it measured

Every change here was kept or dropped on a number. Where a change did not
help, it is listed with the number that showed it, because "we tried that"
is worth as much as "we did that".

Nothing was retrained. Fine-tuning Whisper for Indian lecture speech would
need a labelled corpus of it and GPU-days, and doing it days before a
deadline risks a model that is worse than the one it replaces. This is the
other kind of tuning: how the models are fed, scheduled and run.

## The browser build (`/live`)

Measured by `scripts/bench_live.py`: the whole of `testaudio/lecture_long.wav`
(104 s, 14 sentences) spoken into the page once through Chromium's fake
capture device, then drained. Same machine throughout - an AMD Ryzen laptop
with 12 logical cores and an RDNA-2 integrated GPU.

### Result

| | before | after |
|---|---:|---:|
| Start → listening | 55.3 s cold, 10.9 s warm | **0.1–0.4 s** |
| Start → first caption | 60.5 s cold, 17.0 s warm | **4.1–4.3 s** |
| Inference per caption, mean | 4.0–10.8 s | **1.8–1.9 s** |
| Inference per caption, p95 | 4.7–**52** s | **2.1–2.2 s** |
| Real-time factor | 0.94–3.77 | **0.46–0.50** |
| WER, full lecture | 2.8–3.4 % | 1.1–2.8 % |
| Technical terms kept | 8/8 | 8/8 |

The p95 is the row that matters most. Before, the page had no headroom, so
anything else the laptop was doing pushed it off real time - one run averaged
RTF 3.77. After, it holds between 0.46 and 0.50 run to run.

The first caption now arrives about four seconds after pressing Start, and
most of that is the first sentence itself: it takes four seconds to say.

### What did it

**Preloading (Start: 55 s → 0.4 s).** The model used to download when Start
was pressed. It now starts downloading when the page opens, while a visitor
reads the banner, and Start awaits the same promise. The wait did not go
away - `ready_from_open` is still 11–23 s - it moved to where nobody is
watching a spinner.

**Threads, via cross-origin isolation (inference 2.89× faster).** WebAssembly
is single-threaded unless a page is cross-origin isolated. `vercel.json` now
sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: credentialless`, and ONNX Runtime spreads
each inference over eight threads. Measured on a fixed 6 s clip, alternating
the two conditions across three rounds so drift could not favour either:

| | threads | per clip | RTF |
|---|---:|---:|---:|
| single-threaded | 1 | 5,548 ms | 0.92 |
| isolated | 8 | **1,920 ms** | **0.32** |

`credentialless` rather than `require-corp` because huggingface.co does not
send `Cross-Origin-Resource-Policy`, and `require-corp` would block the model
weights.

**A bug the benchmark caught before it shipped.** Isolation alone produced a
page that never captioned: *"no available backend found. ERR: [wasm]
Failed to fetch dynamically imported module: blob:…"*. The threaded runtime
loads its worker from a `blob:` URL, and the content-security policy did not
allow `blob:` scripts. Every Chrome and Firefox visitor would have hit it -
those are the browsers that honour the isolation header; Safari ignores it
and would have worked, which is exactly how it could have gone unnoticed.
`script-src` now allows `blob:`, and `tests/test_web_build.py` fails if
isolation is ever configured without it.

**Precision per backend.** The page used a 4-bit model on WebGPU. Measured
on the real GPU (the installed Chrome in headless mode reaches it; the
browser Playwright ships does not, so every earlier run had only ever
tested the WASM path):

| | RTF | WER |
|---|---:|---:|
| WebGPU, 4-bit throughout | 0.78 | 3.4 % |
| **WebGPU, fp32 encoder + 4-bit decoder** | **0.46** | 1.1 % |
| **WASM ×8, int8** | **0.48** | 2.2–2.8 % |
| WASM ×8, fp32 encoder + int8 decoder | 0.52 | 3.9 % |

A 4-bit encoder is slow on the GPU as well as lossy: dequantising it costs
more than it saves. On the CPU, int8 is the fast path and fp32 only adds
download. **Read the WER column with care:** over this 180-word lecture one
word is 0.55 %, so the whole spread is two to five words, and no
configuration lost a technical term. The speed column is the one to trust.

**Pre-roll (robustness).** An energy gate fires after the first consonant of
a word, and an earlier deployed run showed captions starting *"morning
everyone"* and *"matrix is a linear transformation"*. The page now keeps
320 ms from before speech starts, as the desktop segmenter already did. It
is intermittent, so no single run proves it; it did not cost WER.

**Merging a backlog instead of dropping it.** Whisper's encoder processes a
fixed 30 s window whatever it is given, so two queued 3 s sentences cost two
encoder passes apart and one together. When inference falls behind, the
page now transcribes the backlog in one call rather than discarding the
oldest segments. With the fixes above it no longer falls behind on this
machine - every run reports 0 merged, 0 dropped - so this is the safety net
for slower hardware, not a speed-up measured here.

**No `chunk_length_s`.** It routed every call through the long-audio
chunking path. A segment here is never longer than 27 s, which already fits
the window whole.

## The desktop pipeline

Measured by `scripts/bench_e2e.py`: the same 104 s lecture played through
the real pipeline - VAD, Whisper Small, NLLB-200 translating into Hindi, and
the Llama 3.2 1B glossary model running alongside, which is how the product
actually runs on a CPU.

### Result

| | old | new |
|---|---:|---:|
| Start → first caption | 10.5 s | **5.6 s** |
| Caption lag, mean | 53.3 s | **18.0 s** |
| Caption lag, worst | 102.9 s | **34.2 s** |
| Backlog when the lecture ends | 101.3 s | **24.4 s** |
| WER | 0.0 % | **0.0 %** |
| Technical terms kept | 8/8 | **8/8** |

*Caption lag* is end of the spoken sentence to its caption on screen - the
number a reader experiences. The old pipeline fell about four seconds
further behind with every sentence and never recovered: by the end of a
104 s lecture, captions were two minutes late. An hour-long lecture would
have been unreadable.

The transcript is word-for-word identical. **The trade:** the Hindi
translation now trails its English caption by 11 s rather than 5 s. English
is what is read live; translation catches up at the next pause.

It still lags 18 s on average, because three models share one CPU. That
residue is the NPU argument - [CONCURRENCY.md](CONCURRENCY.md) - and no
amount of scheduling makes it go away. What scheduling can do is stop a CPU
machine from falling irrecoverably behind, and it now does.

### What did it

**Translation on its own thread.** A sentence used to pay transcription
(6.7 s under load) *then* translation (4.8 s) before the next sentence
started - 11.5 s of work per sentence, arriving every 7.4 s. Translation now
runs on its own thread, so the next English caption only waits for the
transcription in front of it. `translate.concurrent` switches it back.

**Merging a backlog.** When transcription does fall behind, queued segments
are folded into one call - up to 27 s, inside Whisper's 30 s window - rather
than transcribed one by one. Every call pays for a full 30 s encoder pass
however short its audio, so this is close to free throughput: 7 of 18
segments were merged in this run. Keeping up, the queue holds one segment
and nothing merges. `audio.merge_backlog_s = 0` switches it off.

Both are measured together here; `--serial` and `--no-merge` reproduce the
old behaviour for comparison.

## Found along the way

**Forced cuts split words.** A lecturer who talks for twelve seconds without
a 700 ms pause is normal, and the segment cap then cut wherever the clock
landed. A comment said the tail was carried into the next segment; the code
did not do it, so the split word was lost from both captions and consecutive
segments overlapped in time (12.10 s → 11.93 s). The cut now moves to the
quietest 32 ms frame in the last 1.5 s - almost always a gap between words -
and the remainder starts the next segment. `tests/test_vad.py` has three
tests for it; two of them fail against the old code.

## Not done, and why

**Conditioning Whisper on the previous caption.** Standard in long-form
transcription, and it tends to keep technical terms spelled consistently.
Not done because the test audio is synthesised speech already at 1–3 % WER,
so any gain would be unmeasurable here, and an unmeasured change to the
decoder days before a deadline is the wrong trade. It is the first thing to
try once there is real lecture audio with a transcript.

**A lower pause threshold.** 700 ms → 500 ms would bring each caption about
200 ms sooner, against 2–3 s of inference. It would also split sentences at
commas, which costs accuracy and, because every call pays for a full 30 s
encoder window, costs speed too. Not worth it.

**Retraining any model.** Covered at the top.

## Reproducing this

```powershell
python scripts/bench_live.py --runs 2                      # full lecture, WASM
python scripts/bench_live.py --runs 1 --channel chrome     # full lecture, real GPU
python scripts/bench_live.py --inference 4 --rounds 3      # threads, isolated
python scripts/bench_e2e.py                                # desktop pipeline
python scripts/bench_e2e.py --serial --no-merge            # desktop, the old way
```

A laptop someone is using is a noisy place to benchmark. One run here
averaged RTF 5.6 because the machine went to sleep mid-inference; its p95
was 6.8 s, which is how that was spotted. Run twice, and read the p95.
