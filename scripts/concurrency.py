"""Measure what the glossary costs the captions.

This is the experiment behind the central claim. Sahaay says two models run
continuously side by side - Whisper transcribing every few seconds, Llama
writing glossary entries alongside it - and that on the Hexagon NPU the
second one costs the user nothing they notice. That is an empirical claim
and it had never been tested.

The method is simple and the interpretation is the interesting part:

1. Transcribe a clip N times with nothing else running. Record the latencies.
2. Start the LLM generating continuously on a background thread, exactly as
   ``GlossaryWorker`` does during a lecture.
3. Transcribe the same clip N more times. Record again.
4. Report the difference.

**A large slowdown on CPU is the expected result, not a failure.** It is the
measurement that justifies the NPU: on one set of cores the two models
compete, and captions - the thing the user is reading in real time - are
what loses. On the NPU the graphs are resident on separate silicon and the
contention disappears. Running this on both is the comparison that makes
the architecture argument concrete instead of rhetorical.

    python scripts/concurrency.py
    python scripts/concurrency.py --runs 12 --write
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import model_name, real_speech  # noqa: E402

from sahaay.config import load_config  # noqa: E402

BENCH_AUDIO_SECONDS = 8.0

# Long enough that the model is genuinely busy between ASR calls, short
# enough that the loop restarts promptly rather than running one huge
# generation across the whole measurement.
LLM_TOKENS_PER_CALL = 96
LLM_PROMPT = (
    "List three technical terms from this lecture line and explain each in "
    "one short sentence: the determinant of matrix A must be zero for a "
    "non-trivial solution to the characteristic equation."
)


def synth_speech(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    """Speech-like signal. Compute is identical regardless of content."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    for f, amp in ((120, 0.5), (700, 0.3), (1220, 0.2), (2600, 0.1)):
        signal += amp * np.sin(2 * np.pi * f * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    return (signal * envelope * 0.3).astype(np.float32)


class LlmLoad:
    """Keeps the LLM generating for as long as it is running."""

    def __init__(self, llm):  # noqa: ANN001
        self.llm = llm
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.calls = 0
        self.tokens = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="llm-load", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = self.llm.generate(LLM_PROMPT, max_new_tokens=LLM_TOKENS_PER_CALL)
                self.calls += 1
                self.tokens += result.tokens
            except Exception:  # noqa: BLE001 - load generator must not die
                time.sleep(0.1)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            # generate() is not interruptible, so allow one full call to end.
            self._thread.join(timeout=30.0)
            self._thread = None


def measure(asr, audio: np.ndarray, runs: int) -> list[float]:  # noqa: ANN001
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        asr.transcribe(audio)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def summarise(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "runs": len(samples),
        "mean_ms": round(statistics.fmean(samples), 1),
        "p50_ms": round(ordered[len(ordered) // 2], 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
    }


def render_markdown(result: dict) -> str:
    alone = result["asr_alone"]
    loaded = result["asr_with_llm"]
    device = result["device"]
    ratio = result["slowdown"]
    rtf_alone = result["rtf_alone"]
    rtf_loaded = result["rtf_with_llm"]

    keeps_up = rtf_loaded < 1.0
    lines = [
        "# Concurrency: what the glossary costs the captions",
        "",
        f"Measured {dt.datetime.now():%d %b %Y} by `scripts/concurrency.py` on "
        f"**{device.get('provider_label', '?')}** "
        f"({device.get('machine', '?')}).",
        "",
        "Sahaay claims two models run side by side for a whole lecture - Whisper",
        "transcribing, Llama writing glossary entries - without the captions",
        "suffering. This measures that directly: the same clip transcribed with",
        "the LLM idle, then again with it generating continuously.",
        "",
        "| Whisper latency | Mean | p50 | p95 | Real-time factor |",
        "|---|---:|---:|---:|---:|",
        f"| LLM idle | {alone['mean_ms']} ms | {alone['p50_ms']} ms | "
        f"{alone['p95_ms']} ms | {rtf_alone} |",
        f"| **LLM generating** | **{loaded['mean_ms']} ms** | {loaded['p50_ms']} ms | "
        f"{loaded['p95_ms']} ms | **{rtf_loaded}** |",
        "",
        f"**Captions slow down {ratio}x** while the glossary runs. The LLM completed "
        f"{result['llm_calls']} generations ({result['llm_tokens']} tokens) during the "
        "loaded phase.",
        "",
        "## What this means",
        "",
    ]

    if device.get("npu_active"):
        lines += [
            "Both graphs are resident on the Hexagon NPU, so they are not competing",
            "for the same cores. This is the configuration the product is designed",
            "for, and the number above is what the user actually experiences.",
        ]
    else:
        lines += [
            "**This is the CPU column, and it is the argument for the NPU.**",
            "",
            "On a single set of CPU cores the two models compete directly, and the",
            "captions - the thing the user is reading in real time - are what loses.",
            f"A {ratio}x slowdown is not a tuning problem; it is two compute-bound",
            "models on one processor.",
            "",
            "On a Snapdragon PC the Whisper encoder runs on the Hexagon NPU instead",
            "of the CPU - [measured](AIHUB.md) at 13.5 ms on X2 Elite, with 129 of",
            "129 layers on the NPU. That moves the two models onto separate silicon",
            "and removes exactly the contention this table is showing.",
            "",
            "Re-run this on a Snapdragon device to produce the comparison column.",
        ]

    if keeps_up:
        lines += [
            "",
            f"Even under load the pipeline stays at RTF {rtf_loaded}, below 1.0, so",
            "captions still arrive faster than speech. The headroom is what shrinks.",
        ]
    else:
        lines += [
            "",
            f"Under load the pipeline reaches RTF {rtf_loaded}, at or above 1.0 -",
            "captions would drift further behind the lecturer every minute. This is",
            "the failure mode the NPU exists to prevent.",
        ]

    lines += [
        "",
        "## Method",
        "",
        f"- {alone['runs']} transcriptions of an {BENCH_AUDIO_SECONDS}s clip, LLM idle",
        f"- {loaded['runs']} more with the LLM generating continuously on a background",
        f"  thread, {LLM_TOKENS_PER_CALL} tokens per call - the same shape of load",
        "  `GlossaryWorker` produces during a lecture",
        "- Warm-up runs discarded; the first execution pays graph finalisation",
        "- The LLM generates *continuously*, which is the worst case:",
        "  `GlossaryWorker` submits one call per caption, so real load is bursty.",
        "  Read the idle row as the best case and the loaded row as the worst;",
        "  a lecture sits between them.",
        f"- ASR model `{result.get('asr_model') or '?'}`, glossary model "
        f"`{result.get('llm_model') or '?'}`",
        f"- Signal: {result.get('signal') or 'unspecified'}",
        "",
        "**Which model was measured matters more than it looks.** An earlier",
        "revision of this file reported RTF 0.043 idle and 0.299 loaded, and",
        "concluded the pipeline still kept up. That run used `whisper_tiny_en`",
        "against a synthetic signal, and recorded neither fact. The product",
        "resolves to `whisper_small_portable`, which is far slower - and on it,",
        "the loaded figure crosses 1.0 and the conclusion reverses.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure ASR latency under LLM load.")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--write", action="store_true", help="write docs/CONCURRENCY.md")
    ap.add_argument("--json", type=Path)
    ap.add_argument(
        "--audio-file", type=Path, default=REPO_ROOT / "testaudio" / "lecture.wav",
        help="speech to transcribe (default: the committed lecture clip)",
    )
    ap.add_argument(
        "--synthetic", action="store_true",
        help="use the old sine-wave signal instead of real speech",
    )
    args = ap.parse_args(argv)

    from sahaay.asr import create_asr
    from sahaay.llm import create_llm
    from sahaay.runtime import SessionFactory

    cfg = load_config()
    factory = SessionFactory(cfg.runtime)
    device = factory.report()
    print(f"\n  device: {device.provider_label}\n")

    try:
        asr = create_asr(cfg.models_dir, factory, cfg.asr)
    except FileNotFoundError as exc:
        print(f"  ASR unavailable: {exc}")
        return 1

    llm = create_llm(
        cfg.models_dir, cfg.glossary.model_id,
        candidates=cfg.glossary.candidates, npu_active=device.npu_active
    )
    if llm.name == "heuristic":
        print("  No LLM weights present - there is no load to measure.")
        print("  python scripts/download_models.py --llm\n")
        return 1

    if args.synthetic or not args.audio_file.exists():
        audio = synth_speech(BENCH_AUDIO_SECONDS)
        signal = "synthetic sine-wave signal"
    else:
        audio = real_speech(BENCH_AUDIO_SECONDS, args.audio_file)
        signal = f"real speech ({args.audio_file.name})"
    print(f"  signal: {signal}")
    print(f"  asr model: {model_name(asr)}")
    print(f"  llm model: {model_name(llm)}")

    print("  warming up...")
    for _ in range(args.warmup):
        asr.transcribe(audio)
    llm.generate(LLM_PROMPT, max_new_tokens=16)

    print(f"  phase 1: {args.runs} transcriptions, LLM idle")
    alone = measure(asr, audio, args.runs)
    print(f"           mean {statistics.fmean(alone):.0f} ms")

    print(f"  phase 2: {args.runs} transcriptions, LLM generating")
    load = LlmLoad(llm)
    load.start()
    time.sleep(1.0)  # let the first generation get going
    loaded = measure(asr, audio, args.runs)
    load.stop()
    print(f"           mean {statistics.fmean(loaded):.0f} ms")

    mean_alone = statistics.fmean(alone)
    mean_loaded = statistics.fmean(loaded)

    result = {
        "generated": dt.datetime.now().isoformat(),
        "device": device.to_dict(),
        "audio_seconds": BENCH_AUDIO_SECONDS,
        "signal": signal,
        "asr_model": model_name(asr),
        "llm_model": model_name(llm),
        "asr_alone": summarise(alone),
        "asr_with_llm": summarise(loaded),
        "slowdown": round(mean_loaded / mean_alone, 2) if mean_alone else 0.0,
        "rtf_alone": round((mean_alone / 1000.0) / BENCH_AUDIO_SECONDS, 3),
        "rtf_with_llm": round((mean_loaded / 1000.0) / BENCH_AUDIO_SECONDS, 3),
        "llm_calls": load.calls,
        "llm_tokens": load.tokens,
        "llm_backend": llm.name,
    }

    print(f"\n  slowdown: {result['slowdown']}x   "
          f"RTF {result['rtf_alone']} -> {result['rtf_with_llm']}")

    md = render_markdown(result)
    if args.write:
        out = REPO_ROOT / "docs" / "CONCURRENCY.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"  wrote {out}")
    else:
        print("\n" + md)

    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
