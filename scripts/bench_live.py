"""Measure the browser build end to end: how fast, how accurate, what it drops.

check_live.py answers "does it work". This answers "how well", with the
numbers a visitor actually feels:

  ready        Start pressed -> listening. The model download and warm-up.
  first        Start pressed -> first caption on screen.
  inference    Per-caption transcription time, mean and p95.
  RTF          Inference time over audio length. Above 1.0 it falls behind.
  WER          Word error rate over the whole lecture against its script.
  terms        Technical terms that survived, which is the metric the product
               actually depends on.
  dropped      Segments thrown away because inference could not keep up.
               Invisible in the UI; a sentence simply never appears.

The whole of testaudio/lecture_long.wav is spoken into the page once through
Chromium's fake capture device, then the harness waits for the queue to
drain. Every number comes from window.sahaayLive, which live.js fills in and
never reads back.

    python scripts/bench_live.py
    python scripts/bench_live.py --base https://sahaay-offline.vercel.app
    python scripts/bench_live.py --runs 2 --label "after pre-roll"

Run it before and after a change. A browser is noisy; two runs is the least
that tells you whether a difference is real.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_live import WEB_DIR, serve  # noqa: E402
from make_lecture_audio import LECTURE  # noqa: E402

DEFAULT_WAV = REPO_ROOT / "testaudio" / "lecture_long.wav"

TERMS = (
    "eigenvalue", "eigenvector", "determinant", "characteristic",
    "diagonalize", "diagonalization", "symmetric", "matrix",
)

# Spelling a recogniser is free to choose. Counting "center" as wrong for
# "centre" would be measuring a style guide, not a model.
SAME = {"centre": "center", "diagonalise": "diagonalize", "diagonalisation": "diagonalization"}


def words(text: str) -> list[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return [SAME.get(w, w) for w in cleaned.split()]


def wer(reference: str, hypothesis: str) -> float:
    r, h = words(reference), words(hypothesis)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if r[i - 1] == h[j - 1] else 1),
            )
        prev = cur
    return 100.0 * prev[len(h)] / max(len(r), 1)


def terms_kept(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = " ".join(words(reference)), " ".join(words(hypothesis))
    wanted = [t for t in TERMS if t in ref]
    return sum(1 for t in wanted if t in hyp), len(wanted)


def wait_until(page, predicate: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if page.evaluate(predicate):
            return
        time.sleep(0.5)
    raise TimeoutError(f"timed out after {timeout_s:.0f}s waiting for {predicate}")


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def one_run(pw, base: str, wav: Path, settle: float, channel: str | None = None) -> dict:
    duration = wav_seconds(wav)
    # channel="chrome" is the installed Chrome in its new headless mode. The
    # headless_shell Playwright ships has no GPU adapter at all, so it can
    # only ever test the WASM path - and a visitor on a laptop with a GPU
    # takes the WebGPU one.
    browser = pw.chromium.launch(
        headless=True,
        **({"channel": channel} if channel else {}),
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            # %noloop: speak the lecture once, so "the audio has ended" is a
            # moment we can wait for rather than a guess.
            f"--use-file-for-fake-audio-capture={wav}%noloop",
            "--autoplay-policy=no-user-gesture-required",
        ],
    )
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{base}/live/", wait_until="networkidle")
    page.wait_for_selector("#toggle:not([disabled])", timeout=60_000)
    page.click("#toggle")

    # Wait for listening, then for the lecture to be spoken, then for the
    # queue to go quiet. The fake device starts at getUserMedia, i.e. just
    # before "listening", so audio ends ~duration after readyAt.
    # Poll from Python rather than wait_for_function: that evaluates a
    # string inside the page, and the production CSP - which the local
    # server now sends too - rightly forbids eval. Same trap shoot_web.py
    # fell into once.
    wait_until(page, "() => !!(window.sahaayLive && window.sahaayLive.readyAt > 0)", 600)
    time.sleep(duration)

    last, quiet_since = -1, time.time()
    while time.time() - quiet_since < settle:
        n = page.evaluate("window.sahaayLive.captions.length")
        if n != last:
            last, quiet_since = n, time.time()
        time.sleep(1.0)

    m = page.evaluate("JSON.parse(JSON.stringify(window.sahaayLive))")
    crossiso = page.evaluate("self.crossOriginIsolated === true")
    threads = page.evaluate("navigator.hardwareConcurrency || 0")
    badge = page.inner_text("#device-label")
    browser.close()

    caps = m["captions"]
    text = " ".join(c["text"] for c in caps)
    reference = " ".join(LECTURE)
    lat = [c["latency_ms"] for c in caps]
    rtf = [c["latency_ms"] / 1000 / c["audio_s"] for c in caps if c["audio_s"] > 0]
    kept, total = terms_kept(reference, text)

    return {
        "backend": badge,
        "cross_origin_isolated": crossiso,
        "cores": threads,
        "ready_ms": round(m["readyAt"] - m["clickAt"]),
        # performance.now() counts from navigation, so readyAt alone is the
        # wait from opening the page. Preloading moves time out of the column
        # above and into this one; the total is what a visitor who clicks
        # immediately actually waits.
        "ready_from_open_ms": round(m["readyAt"]),
        "first_ms": round(caps[0]["at"] - m["clickAt"]) if caps else None,
        "captions": len(caps),
        "dropped": m["dropped"],
        "merged": m.get("merged", 0),
        "inference_mean_ms": round(statistics.fmean(lat)) if lat else None,
        "inference_p95_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1 if len(lat) > 1 else 0]) if lat else None,
        "rtf_mean": round(statistics.fmean(rtf), 2) if rtf else None,
        "wer": round(wer(reference, text), 1),
        "terms_kept": f"{kept}/{total}",
        "errors": errors[:3],
        "transcript": text,
    }


def clip_samples(wav: Path, seconds: float = 6.0) -> list[float]:
    """The first few seconds of the lecture, as floats the page can take."""
    import array

    with wave.open(str(wav), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(int(seconds * rate))
    pcm = array.array("h", frames)
    return [v / 32768.0 for v in pcm]


def inference_bench(pw, wav: Path, rounds: int, runs: int) -> None:
    """Inference alone, isolated vs not, alternating to cancel drift.

    Alternating matters on a machine someone is using: run all of one
    condition then all of the other and a background task that starts
    halfway through lands entirely on one side.
    """
    samples = clip_samples(wav)
    audio_s = len(samples) / 16000
    table: dict[str, list[float]] = {"single-threaded": [], "isolated": []}
    meta: dict[str, dict] = {}

    with serve(WEB_DIR, production_headers=False) as bare, \
            serve(WEB_DIR, production_headers=True) as iso:
        for r in range(rounds):
            for label, base in (("single-threaded", bare), ("isolated", iso)):
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"{base}/live/", wait_until="domcontentloaded")
                wait_until(page, "() => typeof window.sahaayBench === 'function'", 60)
                out = page.evaluate(
                    "([s, n]) => window.sahaayBench(s, n)", [samples, runs + 1]
                )
                browser.close()
                timed = out["times"][1:]   # the first run pays graph setup
                table[label] += timed
                meta[label] = out
                print(f"    round {r + 1} {label:<16} threads {out['threads']}  "
                      f"mean {statistics.fmean(timed):7.0f} ms", flush=True)

    print(f"\n  inference on {audio_s:.0f} s of lecture, {rounds} rounds x {runs} runs")
    for label, times in table.items():
        m = meta[label]
        print(f"    {label:<16} isolated={m['isolated']!s:<5} threads={m['threads']:<2} "
              f"mean {statistics.fmean(times):7.0f} ms   median {statistics.median(times):7.0f} ms   "
              f"RTF {statistics.median(times) / 1000 / audio_s:.2f}")
    a = statistics.median(table["single-threaded"])
    b = statistics.median(table["isolated"])
    print(f"\n    speed-up from isolation: {a / b:.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inference", type=int, metavar="N", default=0,
        help="time N transcriptions of a fixed clip, isolated vs not, instead",
    )
    ap.add_argument("--rounds", type=int, default=3, help="with --inference")
    ap.add_argument(
        "--channel", help='browser to drive, e.g. "chrome" for the installed Chrome (has a GPU)',
    )
    ap.add_argument("--base", help="measure a deployment instead of the local web/")
    ap.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--settle", type=float, default=25.0, help="seconds of no new captions = done")
    ap.add_argument("--label", default="", help="name this run in the output")
    ap.add_argument("--json", type=Path, help="append results here")
    ap.add_argument(
        "--bare", action="store_true",
        help="serve without vercel.json's headers (no CSP, no isolation)",
    )
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    if args.inference:
        with sync_playwright() as pw:
            inference_bench(pw, args.wav, args.rounds, args.inference)
        return 0

    source = (
        contextlib.nullcontext(args.base.rstrip("/")) if args.base
        else serve(WEB_DIR, production_headers=not args.bare)
    )
    results = []
    with source as base, sync_playwright() as pw:
        for i in range(args.runs):
            print(f"  run {i + 1}/{args.runs} against {base}/live/ ...", flush=True)
            results.append(one_run(pw, base, args.wav, args.settle, args.channel))

    keys = ["backend", "cross_origin_isolated", "cores", "ready_ms", "ready_from_open_ms",
            "first_ms", "captions", "dropped", "merged", "inference_mean_ms",
            "inference_p95_ms", "rtf_mean", "wer", "terms_kept"]
    print(f"\n  {args.label or 'result'}")
    for k in keys:
        vals = "   ".join(str(r[k]) for r in results)
        print(f"    {k:<22} {vals}")
    print("\n  transcript (run 1):")
    print("    " + results[0]["transcript"][:600])
    for r in results:
        if r["errors"]:
            print(f"\n  page errors: {r['errors']}")

    if args.json:
        existing = json.loads(args.json.read_text()) if args.json.exists() else []
        existing.append({"label": args.label, "runs": results})
        args.json.write_text(json.dumps(existing, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
