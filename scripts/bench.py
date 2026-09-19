"""Benchmark harness - produces docs/BENCHMARKS.md.

The numbers in the README come from this script, run on the machine named in
the output. Nothing is typed in by hand.

What it measures, per execution provider:

* **mel**       - the NumPy front end (CPU always; included so the ASR figure
                  can be read as encoder+decoder rather than everything)
* **asr**       - Whisper on a fixed audio clip, warm
* **translate** - one caption line through NLLB
* **llm**       - one glossary lookup, and the tokens/sec that implies

Warm-up runs are discarded. The first execution on the HTP pays graph
finalisation, which can be seconds, and reporting that as steady-state
latency would flatter CPU by a wide margin - in the wrong direction.

    python scripts/bench.py                      # current provider
    python scripts/bench.py --compare            # QNN vs CPU, same process
    python scripts/bench.py --write              # update docs/BENCHMARKS.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahaay.config import SUPPORTED_LANGUAGES, load_config  # noqa: E402
from sahaay.features import log_mel_spectrogram  # noqa: E402

BENCH_AUDIO_SECONDS = 8.0
SAMPLE_LINE = "So today we will start with eigenvalues and eigenvectors of a symmetric matrix."


@dataclass
class Result:
    stage: str
    provider: str
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "provider": self.provider,
            "runs": self.runs,
            "mean_ms": round(self.mean_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "min_ms": round(self.min_ms, 1),
            **self.extra,
        }


def summarise(stage: str, provider: str, samples: list[float], **extra) -> Result:
    ordered = sorted(samples)
    return Result(
        stage=stage,
        provider=provider,
        runs=len(samples),
        mean_ms=statistics.fmean(samples),
        p50_ms=ordered[len(ordered) // 2],
        p95_ms=ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        min_ms=ordered[0],
        extra=extra,
    )


def synth_speech(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    """A speech-like test signal.

    Not real speech, and it is not meant to be: this measures *compute*, and
    the encoder does identical work regardless of what was said. Word error
    rate needs a real corpus and is out of scope for a latency harness - the
    README says so rather than implying these numbers cover accuracy.
    """
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    # A 120 Hz glottal pulse train with three formants and syllable-rate
    # amplitude modulation - roughly the spectral envelope of voiced speech.
    signal = np.zeros_like(t)
    for f, amp in ((120, 0.5), (700, 0.3), (1220, 0.2), (2600, 0.1)):
        signal += amp * np.sin(2 * np.pi * f * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)  # ~4 syllables/sec
    return (signal * envelope * 0.3).astype(np.float32)


def timeit(fn, runs: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def bench_provider(provider: str | None, runs: int, warmup: int) -> tuple[list[Result], dict]:
    """Run every stage on one provider. Missing models are skipped, not faked."""
    from sahaay.asr import create_asr
    from sahaay.llm import create_llm
    from sahaay.runtime import SessionFactory
    from sahaay.translate import create_translator

    cfg = load_config()
    if provider:
        cfg.runtime.provider_priority = (
            [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
        )

    factory = SessionFactory(cfg.runtime)
    report = factory.report()
    label = report.provider_label
    print(f"\n  == {label} ==")

    audio = synth_speech(BENCH_AUDIO_SECONDS)
    results: list[Result] = []

    # mel -- always CPU, reported so asr can be read as model-only time.
    samples = timeit(lambda: log_mel_spectrogram(audio), runs, warmup)
    results.append(summarise("mel (CPU)", label, samples))
    print(f"     mel          {statistics.fmean(samples):8.1f} ms")

    # asr
    try:
        asr = create_asr(cfg.models_dir, factory, cfg.asr)
        samples = timeit(lambda: asr.transcribe(audio), runs, warmup)
        rtf = (statistics.fmean(samples) / 1000.0) / BENCH_AUDIO_SECONDS
        results.append(summarise("asr", label, samples, rtf=round(rtf, 3),
                                 audio_s=BENCH_AUDIO_SECONDS))
        print(f"     asr          {statistics.fmean(samples):8.1f} ms   RTF {rtf:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"     asr          skipped ({type(exc).__name__}: {exc})")

    # translate
    try:
        tr = create_translator(cfg.models_dir, factory, cfg.translate)
        if type(tr).__name__ == "PassthroughTranslator":
            print("     translate    skipped (weights not downloaded)")
        else:
            samples = timeit(lambda: tr.translate(SAMPLE_LINE, "hi"), runs, warmup)
            results.append(summarise("translate", label, samples, target="hi"))
            print(f"     translate    {statistics.fmean(samples):8.1f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"     translate    skipped ({exc})")

    # llm
    try:
        llm = create_llm(cfg.models_dir, cfg.glossary.model_id)
        if llm.name == "heuristic":
            print("     llm          skipped (weights not downloaded)")
        else:
            prompt = f"<transcript>{SAMPLE_LINE}</transcript>\nList technical terms."
            holder: dict = {}

            def run_llm():
                holder["r"] = llm.generate(prompt, max_new_tokens=64)

            samples = timeit(run_llm, max(3, runs // 3), 1)
            tps = holder["r"].tokens_per_second if holder.get("r") else 0.0
            results.append(summarise("llm (glossary)", label, samples,
                                     tokens_per_second=round(tps, 1),
                                     backend=llm.name))
            print(f"     llm          {statistics.fmean(samples):8.1f} ms   {tps:.1f} tok/s")
    except Exception as exc:  # noqa: BLE001
        print(f"     llm          skipped ({exc})")

    return results, report.to_dict()


def render_markdown(all_results: list[Result], devices: list[dict], notes: str) -> str:
    now = dt.datetime.now().strftime("%d %b %Y")
    primary = devices[0] if devices else {}

    lines = [
        "# Benchmarks",
        "",
        f"Generated by `scripts/bench.py` on {now}. Every number here was measured",
        "on the machine described below - none are copied from a datasheet.",
        "",
        "## Machine",
        "",
        f"- **Architecture** {primary.get('machine', '?')}"
        f"{' (ARM64)' if primary.get('is_arm64') else ''}",
        f"- **Processor** {primary.get('processor', '?')}",
        f"- **ONNX Runtime** {primary.get('ort_version', '?')}",
        f"- **Providers available** {', '.join(primary.get('available_providers', [])) or 'none'}",
        f"- **Python** {platform.python_version()}",
        "",
        "## Results",
        "",
        "| Stage | Provider | Runs | Mean | p50 | p95 | Min | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for r in all_results:
        extra_bits = []
        if "rtf" in r.extra:
            extra_bits.append(f"RTF {r.extra['rtf']}")
        if "tokens_per_second" in r.extra:
            extra_bits.append(f"{r.extra['tokens_per_second']} tok/s")
        if "audio_s" in r.extra:
            extra_bits.append(f"{r.extra['audio_s']}s audio")
        lines.append(
            f"| {r.stage} | {r.provider} | {r.runs} | {r.mean_ms:.1f} ms | "
            f"{r.p50_ms:.1f} ms | {r.p95_ms:.1f} ms | {r.min_ms:.1f} ms | "
            f"{', '.join(extra_bits)} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "**RTF** (real-time factor) is processing time divided by audio duration.",
        "Below 1.0 means the pipeline transcribes faster than speech arrives, which",
        "is the bar a live captioner has to clear. Above 1.0 and captions drift",
        "further behind the lecturer every minute.",
        "",
        "**p95, not just mean.** A captioner with a 200 ms mean and a 3 s p95 feels",
        "broken, because the p95 is the one the user notices mid-sentence.",
        "",
        "**Warm-up runs are discarded.** The first execution on the Hexagon NPU pays",
        "graph finalisation, which can take seconds. Reporting that as steady-state",
        "latency would flatter the CPU column substantially.",
        "",
        "**These are latency numbers, not accuracy numbers.** The harness uses a",
        "synthetic speech-like signal, because the encoder does identical compute",
        "regardless of what was said. Word error rate needs a real labelled corpus",
        "and is not measured here.",
        "",
    ]
    if notes:
        lines += ["## Notes", "", notes, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark Sahaay's stages.")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--provider", help="force one execution provider")
    ap.add_argument("--compare", action="store_true",
                    help="measure the preferred provider and CPU, for a side-by-side")
    ap.add_argument("--write", action="store_true", help="update docs/BENCHMARKS.md")
    ap.add_argument("--json", type=Path, help="also write raw results here")
    args = ap.parse_args(argv)

    providers: list[str | None] = [args.provider]
    if args.compare:
        providers = [None, "CPUExecutionProvider"]

    all_results: list[Result] = []
    devices: list[dict] = []
    for p in providers:
        try:
            results, device = bench_provider(p, args.runs, args.warmup)
        except RuntimeError as exc:
            print(f"\n  !! {exc}")
            return 1
        all_results.extend(results)
        devices.append(device)

    if not all_results:
        print("\n  Nothing measured. Download models first:")
        print("    python scripts/download_models.py --auto\n")
        return 1

    notes = ""
    if args.compare and len(devices) > 1 and devices[0]["provider"] == devices[1]["provider"]:
        notes = (
            "`--compare` was requested but only one provider is available on this "
            "machine, so both columns are the same device. Run this on a "
            "Snapdragon PC, or submit the models to the Qualcomm AI Hub device "
            "farm (`scripts/aihub_profile.py`), for genuine NPU-vs-CPU figures."
        )

    md = render_markdown(all_results, devices, notes)
    if args.write:
        out = REPO_ROOT / "docs" / "BENCHMARKS.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n  wrote {out}")
    else:
        print("\n" + md)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"generated": dt.datetime.now().isoformat(),
                 "devices": devices,
                 "results": [r.to_dict() for r in all_results]},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
