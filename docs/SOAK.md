# Soak test: an hour-long lecture

Ran 20 Sep 2026 by `scripts/soak.py` - 60 simulated minutes, 60.5s wall clock, on CPU.

Every other test in this repo runs for seconds. The failure modes of an
hour are different in kind: unbounded lists, queues that only grow, event
history that accumulates, threads that leak. None appear in a short run,
and all of them surface on stage.

## Growth per simulated minute

| Measure | Start | End | Per minute | |
|---|---:|---:|---:|---|
| Resident memory | 57.5 MB | 58.1 MB | +0.01 | flat |
| Captions retained | 0 | 17 | +0.28 | by design |
| Glossary entries | 0 | 2 | +0.04 | by design |
| Event history | 1 | 37 | +0.61 | capped at 500 |
| Metric samples | 0 | 34 | +0.57 | capped at 4000 |
| Segment queue depth | 0 | 0 | +0.00 | flat |
| Threads | 4 | 4 | +0.00 | flat |

## Verdict: pass - nothing grows without bound

Nothing grows without bound. Captions and glossary entries do grow -
they are the transcript, and a student expects to keep it - but the
structures that must not grow do not: event history is capped, metric
buffers are ring buffers, the segment queue stays near empty, and the
thread count is flat across the whole run.

## Method and its limits

Mock mode drives the full pipeline - segmentation, ASR, translation,
glossary, the event bus and the notes writer - on a scripted transcript,
so the plumbing under test is the real plumbing.

What it does **not** exercise is model memory: the mock ASR does not hold
Whisper's KV cache, and the heuristic glossary is not a 3B model. Resident
memory here is the application's own growth, not the total footprint of a
loaded pipeline. Re-run with weights present for that figure.
