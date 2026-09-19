# Concurrency: what the glossary costs the captions

Measured 20 Sep 2026 by `scripts/concurrency.py` on **CPU** (AMD64).

Sahaay claims two models run side by side for a whole lecture - Whisper
transcribing, Llama writing glossary entries - without the captions
suffering. This measures that directly: the same clip transcribed with
the LLM idle, then again with it generating continuously.

| Whisper latency | Mean | p50 | p95 | Real-time factor |
|---|---:|---:|---:|---:|
| LLM idle | 343.7 ms | 352.3 ms | 374.3 ms | 0.043 |
| **LLM generating** | **2390.7 ms** | 2214.3 ms | 3063.9 ms | **0.299** |

**Captions slow down 6.96x** while the glossary runs. The LLM completed 13 generations (312 tokens) during the loaded phase.

## What this means

**This is the CPU column, and it is the argument for the NPU.**

On a single set of CPU cores the two models compete directly, and the
captions - the thing the user is reading in real time - are what loses.
A 6.96x slowdown is not a tuning problem; it is two compute-bound
models on one processor.

On a Snapdragon PC the Whisper encoder runs on the Hexagon NPU ([measured](AIHUB.md): 13.5 ms on X2 Elite, 129 of 129 layers on the
NPU), which is precisely the contention this table shows.

Re-run this on a Snapdragon device to produce the comparison column.

Even under load the pipeline stays at RTF 0.299, below 1.0, so
captions still arrive faster than speech. The headroom is what shrinks.

## Method

- 8 transcriptions of a 8.0s clip, LLM idle
- 8 more with the LLM generating continuously on a background
  thread, 96 tokens per call - the same shape of load
  `GlossaryWorker` produces during a lecture
- Warm-up runs discarded; the first execution pays graph finalisation
