# Architecture

## Threading

Three threads, and the split is the whole reason captions stay live.

| Thread | Job | Must never |
|---|---|---|
| `capture` | Read the audio device, resample to 16 kHz, segment on pauses | Block — blocking here drops real speech |
| `transcribe` | Drain segments, run Whisper, then translation, publish | Fall so far behind that captions leave the live edge |
| `glossary` | Run the LLM at low priority | Interfere with either of the above |

If capture and transcription shared a thread, a single slow segment — a long sentence, or a cold graph on first execution — would gap the audio. Splitting them lets that slowness absorb into a queue instead.

### Both queues drop the oldest item, not the newest

This looks wrong and is deliberate. If the transcriber falls behind, the useful thing to show the student is what the lecturer is saying *now*. Dropping the newest segment would preserve a growing backlog and eventually explain a term four minutes after it was spoken — technically complete, practically useless.

```python
except queue.Full:
    self._segments.get_nowait()      # discard the oldest
    self._segments.put_nowait(seg)
```

The same rule applies to the glossary queue and to WebSocket delivery: a stalled browser tab must never back-pressure the audio pipeline.

## Segmentation

Whisper is a 30-second batch model. Running it on fixed 30 s windows would mean captions arriving half a minute late — technically a transcript, not a live captioner.

So the segmenter cuts on natural pauses:

- Flush after `silence_flush_ms` (default 700 ms) of non-speech
- Force a cut at `max_segment_s` (default 12 s) regardless

The forced cut matters more than it looks. Lecturers who never pause for 700 ms are common, and without the cap the UI would starve indefinitely while the segment grew.

Segments shorter than `min_segment_s` are discarded — a cough, a chair, a keystroke.

`speech_pad_ms` of audio from *before* the VAD triggered is prepended, so a segment doesn't start mid-consonant.

## Why the mel front end is hand-written

`sahaay/features.py` implements Whisper's log-mel spectrogram in ~60 lines of NumPy rather than importing librosa or torchaudio.

Two reasons. Windows ARM64 wheel coverage for both is patchy, and this is the single component where being subtly wrong produces no error at all — the encoder receives out-of-distribution input and emits fluent, confident, wrong text. So it is implemented explicitly and tested against known tone frequencies:

```python
@pytest.mark.parametrize("freq,low,high",
    [(200, 0, 12), (440, 5, 20), (2000, 30, 55), (6000, 60, 79)])
def test_tone_lands_in_expected_mel_band(...)
```

A wrong mel scale moves those peaks and the test fails. Nothing else in the system would have noticed.

## Model packaging is introspected, not configured

Whisper ships in two shapes depending on where you got it:

- **Qualcomm AI Hub** — separate encoder/decoder graphs, w8a16, precompiled QNN context binary, decoder takes one token plus an explicit position `index`
- **HF Optimum** — `encoder_model.onnx` + `decoder_model_merged.onnx`, decoder takes the whole prefix

`asr.py` reads the graph's input names and adapts, rather than branching on a config flag. AI Hub re-exports models periodically and a hard-coded shape assumption is the most likely thing in this repo to rot.

The same applies to `n_mels`: read off the encoder's declared input shape (80, or 128 for large-v3) instead of assumed.

## Term protection

Naive translation of code-mixed lecture speech destroys the terms that matter most. `translate.py` swaps protected spans for sentinels before the NLLB call and restores them after:

```
in:        "Matrix A ka determinant zero hoga, use SVD for 2.5 ms stability."
protected: "Matrix A ka Qx0z zero hoga, use Qx2z for Qx1z stability."
out:       "मैट्रिक्स A का determinant शून्य होगा, SVD का उपयोग 2.5 ms के लिए।"
```

Protected: acronyms (`SVD`, `TOPS`), CamelCase identifiers (`NumPy`), numerics with units (`2.5 ms`), and the seeded technical vocabulary.

Sentinels are short alphanumerics (`Qx0z`) rather than the conventional `__TERM_0__`, because underscore-heavy tokens get shredded by subword tokenisers and come back unrecoverable.

## The degradation ladder

Every stage answers "what if this is missing?" with something useful rather than an exception:

```
ASR        Whisper on NPU → Whisper on CPU → (required; no fallback)
Translate  NLLB → passthrough, and the UI *says* it is passthrough
Glossary   Llama on NPU → Llama on CPU → seeded heuristic
VAD        Silero → adaptive energy gating
Runtime    QNN → DirectML → CPU
Everything --mock, a scripted transcript through the real pipeline
```

The rule is that the UI never claims a capability it does not have. A passthrough translation renders as "translation model not installed", not as English text under a Hindi heading.

## Events

The pipeline is decoupled from the UI by a small pub/sub bus (`bus.py`). Worker threads call `publish()`; the bus hops to the asyncio loop via `call_soon_threadsafe`.

| Event | Meaning |
|---|---|
| `partial` | In-flight placeholder, replaced by `caption` at the same index |
| `caption` | Finalised line + ASR latency and RTF |
| `translation` | Attaches to an existing caption index |
| `gloss` | A jargon term and its explanation |
| `metric` | Per-stage latency snapshot |
| `status` | Device, run state, loading progress |
| `notes` | End-of-session payload |

History is replayed to late subscribers so a browser opening mid-lecture sees the transcript so far — except `partial` and `metric`, which are transient by definition.

## Metrics

One `Metrics` collector backs both the live RTF badge and `docs/BENCHMARKS.md`, so what a reviewer reads is what the app measured.

Percentiles, not just means: a captioner with a 200 ms mean and a 3 s p95 feels broken, because the p95 is the one the user notices mid-sentence. Ring buffers are bounded so a three-hour lecture does not grow memory.

**RTF** (real-time factor) = processing time ÷ audio duration. Below 1.0 the pipeline keeps up with live speech. It is the one number that decides whether the product works at all.
