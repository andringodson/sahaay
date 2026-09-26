"""Measure the desktop pipeline end to end, the way a lecture exercises it.

bench.py times each stage alone. concurrency.py times Whisper with the LLM
hammering it. Neither answers the question a student has: I pressed Start
and the lecturer is talking - how long until I read it, how much is right,
and does it fall further behind as the hour goes on?

This plays testaudio/lecture_long.wav through the real Pipeline - real VAD,
real Whisper, real NLLB, and the glossary model running alongside whenever
its weights are present - and reports:

  load          load_models(), including warm-up
  first         Start -> first caption
  caption lag   end of each spoken sentence -> its caption, mean and max.
                The number a reader actually experiences.
  tail          audio ends -> last caption. Positive and growing means the
                pipeline is not keeping up and a backlog is forming.
  WER / terms   against the script the audio was synthesised from

    python scripts/bench_e2e.py
    python scripts/bench_e2e.py --label "after overlap" --json results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench_live import terms_kept, wer  # noqa: E402
from make_lecture_audio import LECTURE  # noqa: E402

from sahaay.config import load_config  # noqa: E402

WAV = REPO_ROOT / "testaudio" / "lecture_long.wav"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", type=Path, default=WAV)
    ap.add_argument("--language", default="hi")
    ap.add_argument("--no-glossary", action="store_true", help="measure without the LLM running")
    ap.add_argument("--serial", action="store_true", help="translate inline, the old way")
    ap.add_argument("--no-merge", action="store_true", help="never merge a backlog, the old way")
    ap.add_argument("--spin", action="store_true", help="let ORT threads spin-wait (the old default)")
    ap.add_argument("--llm-threads", type=int, help="glossary CPU threads; 0 = unthrottled")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from sahaay.audio import load_wav
    from sahaay.pipeline import Pipeline

    cfg = load_config()
    cfg.audio_file = args.wav
    cfg.audio_rate = 1.0
    cfg.translate.target_language = args.language
    if args.no_glossary:
        cfg.glossary.enabled = False
    if args.serial:
        cfg.translate.concurrent = False
    if args.no_merge:
        cfg.audio.merge_backlog_s = 0
    if args.spin:
        cfg.runtime.thread_spinning = True
    if args.llm_threads is not None:
        cfg.glossary.cpu_threads = args.llm_threads

    duration = load_wav(args.wav, cfg.audio.sample_rate).size / cfg.audio.sample_rate

    pipeline = Pipeline(cfg)
    events: list[tuple[float, str, dict]] = []
    original = pipeline.bus.publish

    def tap(kind, **data):
        event = original(kind, **data)
        if kind in ("caption", "translation", "gloss"):
            events.append((time.monotonic(), kind, data))
        return event

    pipeline.bus.publish = tap

    t0 = time.monotonic()
    pipeline.load_models()
    load_s = time.monotonic() - t0

    started = time.monotonic()
    pipeline.start()

    # Play the file, then wait for the backlog to clear.
    time.sleep(duration)
    audio_end = time.monotonic()
    last, quiet = len(events), time.monotonic()
    while time.monotonic() - quiet < 20.0:
        time.sleep(0.5)
        if len(events) != last:
            last, quiet = len(events), time.monotonic()
    pipeline.stop()

    caps = [(t, d) for t, k, d in events if k == "caption"]
    trans = [(t, d) for t, k, d in events if k == "translation" and not d.get("passthrough")]
    glosses = [d for _, k, d in events if k == "gloss"]
    reference = " ".join(LECTURE)
    text = " ".join(d["text"] for _, d in caps)

    # Lag: end of the spoken sentence -> its caption on screen. The caption
    # event carries start_s and, through latency/rtf, the segment's length,
    # so its end on the file's clock is recoverable; started + that is the
    # same moment on the wall clock.
    lags = []
    for t, d in caps:
        if d.get("rtf"):
            seconds = (d["latency_ms"] / 1000.0) / d["rtf"]
            lags.append(t - (started + d.get("start_s", 0.0) + seconds))
    seg_lag = []
    by_index = {d["index"]: t for t, d in caps}
    for t, d in trans:
        if d["index"] in by_index:
            seg_lag.append(t - by_index[d["index"]])

    asr_ms = [d["latency_ms"] for _, d in caps]
    tr_ms = [d["latency_ms"] for _, d in trans]
    kept, total = terms_kept(reference, text)

    result = {
        "label": args.label,
        "provider": pipeline.device.provider_label,
        "glossary": "off" if args.no_glossary else getattr(pipeline._glossary, "backend", "on"),
        "load_s": round(load_s, 1),
        "first_s": round(caps[0][0] - started, 1) if caps else None,
        "caption_lag_mean_s": round(statistics.fmean(lags), 1) if lags else None,
        "caption_lag_max_s": round(max(lags), 1) if lags else None,
        "captions": len(caps),
        "translations": len(trans),
        "glossary_entries": len(glosses),
        "asr_mean_ms": round(statistics.fmean(asr_ms)) if asr_ms else None,
        "translate_mean_ms": round(statistics.fmean(tr_ms)) if tr_ms else None,
        "caption_to_translation_s": round(statistics.fmean(seg_lag), 1) if seg_lag else None,
        "tail_s": round(caps[-1][0] - audio_end, 1) if caps else None,
        "merged_segments": pipeline.metrics.snapshot().get("merged_segments", 0),
        "wer": round(wer(reference, text), 1),
        "terms_kept": f"{kept}/{total}",
    }

    print(f"\n  {args.label or 'e2e'}   ({duration:.0f}s of lecture)")
    for k, v in result.items():
        if k != "label":
            print(f"    {k:<26} {v}")
    print("\n  transcript:\n    " + text[:700])

    if args.json:
        existing = json.loads(args.json.read_text()) if args.json.exists() else []
        existing.append(result)
        args.json.write_text(json.dumps(existing, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
