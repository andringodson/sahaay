# Concurrency: what the glossary costs the captions

Measured 22 Sep 2026 by `scripts/concurrency.py` on **CPU** (AMD64).

Sahaay claims two models run side by side for a whole lecture - Whisper
transcribing, Llama writing glossary entries - without the captions
suffering. This measures that directly: the same clip transcribed with
the LLM idle, then again with it generating continuously.

| Whisper latency | Mean | p50 | p95 | Real-time factor |
|---|---:|---:|---:|---:|
| LLM idle | 3275.9 ms | 3308.3 ms | 3422.3 ms | 0.409 |
| **LLM generating** | **12400.2 ms** | 12251.1 ms | 13241.6 ms | **1.55** |

**Captions slow down 3.79x** while the glossary runs. The LLM completed 44 generations (1056 tokens) during the loaded phase.

## What this means

**This is the CPU column, and it is the argument for the NPU.**

On a single set of CPU cores the two models compete directly, and the
captions - the thing the user is reading in real time - are what loses.
A 3.79x slowdown is not a tuning problem; it is two compute-bound
models on one processor.

On a Snapdragon PC the Whisper encoder runs on the Hexagon NPU instead
of the CPU - [measured](AIHUB.md) at 13.5 ms on X2 Elite, with 129 of
129 layers on the NPU. That moves the two models onto separate silicon
and removes exactly the contention this table is showing.

Re-run this on a Snapdragon device to produce the comparison column.

Under load the pipeline reaches RTF 1.55, at or above 1.0 -
captions would drift further behind the lecturer every minute. This is
the failure mode the NPU exists to prevent.

## Method

- 8 transcriptions of an 8.0s clip, LLM idle
- 8 more with the LLM generating continuously on a background
  thread, 96 tokens per call - the same shape of load
  `GlossaryWorker` produces during a lecture
- Warm-up runs discarded; the first execution pays graph finalisation
- The LLM generates *continuously*, which is the worst case:
  `GlossaryWorker` submits one call per caption, so real load is bursty.
  Read the idle row as the best case and the loaded row as the worst;
  a lecture sits between them.
- ASR model `whisper_small_portable`, glossary model `cpu-int4-rtn-block-32-acc-level-4`
- Signal: real speech (lecture.wav)

**Which model was measured matters more than it looks.** An earlier
revision of this file reported RTF 0.043 idle and 0.299 loaded, and
concluded the pipeline still kept up. That run used `whisper_tiny_en`
against a synthetic signal, and recorded neither fact. The product
resolves to `whisper_small_portable`, which is far slower - and on it,
the loaded figure crosses 1.0 and the conclusion reverses.
