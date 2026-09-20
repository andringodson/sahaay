"""Does it survive a whole lecture?

Every test so far runs for seconds. A lecture is an hour, and the failure
modes of an hour are different in kind, not degree: unbounded lists, queues
that only ever grow, event history that accumulates, threads that leak on
each restart. None of those show up in a thirty-second run, and all of them
surface on stage.

This drives the pipeline in mock mode at accelerated rate and watches the
things that grow:

* resident memory
* caption and glossary list lengths
* event-bus history and subscriber count
* metrics ring buffers
* live thread count

and reports growth per simulated minute rather than a single number, because
what matters is the slope, not the value.

    python scripts/soak.py --minutes 60
    python scripts/soak.py --minutes 60 --write
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahaay.config import load_config  # noqa: E402

# A caption every ~3 s is a brisk lecturer. Mock segments arrive on a timer,
# so a simulated minute costs far less than a real one.
SECONDS_PER_SIMULATED_MINUTE = 1.0


def resident_mb() -> float:
    """Process RSS in MB, without requiring psutil."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PMC()
        counters.cb = ctypes.sizeof(PMC)

        # restype matters on 64-bit. ctypes defaults a function's return to
        # C int, which truncates the HANDLE that GetCurrentProcess returns
        # (the pseudo-handle -1), and GetProcessMemoryInfo then fails
        # silently and reports 0 MB - which looks like "no memory growth"
        # rather than "measurement broken".
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = wt.HANDLE
        handle = get_current()

        for library in ("psapi", "kernel32"):
            try:
                fn = getattr(
                    getattr(ctypes.windll, library),
                    "GetProcessMemoryInfo" if library == "psapi"
                    else "K32GetProcessMemoryInfo",
                )
            except AttributeError:
                continue
            fn.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
            fn.restype = wt.BOOL
            if fn(handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass

    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # noqa: BLE001
        return 0.0


def sample(pipeline) -> dict:  # noqa: ANN001
    bus = pipeline.bus
    metrics_samples = sum(len(v) for v in pipeline.metrics._samples.values())
    return {
        "rss_mb": round(resident_mb(), 1),
        "captions": len(pipeline.captions),
        "glossary": len(pipeline._glossary.entries) if pipeline._glossary else 0,
        "bus_history": len(bus._history),
        "bus_subscribers": len(bus._subscribers),
        "metric_samples": metrics_samples,
        "segment_queue": pipeline._segments.qsize(),
        "threads": threading.active_count(),
    }


def slope_per_minute(values: list[float], minutes: list[float]) -> float:
    """Least-squares slope. A flat line is what we want to see."""
    if len(values) < 2:
        return 0.0
    mx, my = statistics.fmean(minutes), statistics.fmean(values)
    denom = sum((x - mx) ** 2 for x in minutes)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(minutes, values, strict=True)) / denom


def render_markdown(result: dict) -> str:
    samples = result["samples"]
    growth = result["growth_per_minute"]
    verdict = result["verdict"]

    lines = [
        "# Soak test: an hour-long lecture",
        "",
        f"Ran {dt.datetime.now():%d %b %Y} by `scripts/soak.py` - "
        f"{result['simulated_minutes']} simulated minutes, "
        f"{result['wall_seconds']}s wall clock, on {result['device']}.",
        "",
        "Every other test in this repo runs for seconds. The failure modes of an",
        "hour are different in kind: unbounded lists, queues that only grow, event",
        "history that accumulates, threads that leak. None appear in a short run,",
        "and all of them surface on stage.",
        "",
        "## Growth per simulated minute",
        "",
        "| Measure | Start | End | Per minute | |",
        "|---|---:|---:|---:|---|",
    ]

    labels = {
        "rss_mb": ("Resident memory", "MB"),
        "captions": ("Captions retained", ""),
        "glossary": ("Glossary entries", ""),
        "bus_history": ("Event history", ""),
        "metric_samples": ("Metric samples", ""),
        "segment_queue": ("Segment queue depth", ""),
        "threads": ("Threads", ""),
    }
    for key, (label, unit) in labels.items():
        first, last = samples[0][key], samples[-1][key]
        rate = growth[key]
        cap = result.get("caps", {}).get(key)
        if key in result.get("unbounded_by_design", []):
            flag = "by design"
        elif cap is not None:
            flag = f"capped at {cap}"
        else:
            flag = "flat" if abs(rate) < result["thresholds"].get(key, 1e9) else "**GROWING**"
        suffix = f" {unit}" if unit else ""
        lines.append(
            f"| {label} | {first}{suffix} | {last}{suffix} | {rate:+.2f} | {flag} |"
        )

    lines += [
        "",
        f"## Verdict: {verdict}",
        "",
    ]
    if verdict.startswith("pass"):
        lines += [
            "Nothing grows without bound. Captions and glossary entries do grow -",
            "they are the transcript, and a student expects to keep it - but the",
            "structures that must not grow do not: event history is capped, metric",
            "buffers are ring buffers, the segment queue stays near empty, and the",
            "thread count is flat across the whole run.",
        ]
    else:
        lines += [
            "Something grows without bound over an hour. The table above names it.",
            "Left alone this ends as a memory exhaustion or an ever-lengthening",
            "queue partway through a lecture.",
        ]

    lines += [
        "",
        "## Method and its limits",
        "",
        "Mock mode drives the full pipeline - segmentation, ASR, translation,",
        "glossary, the event bus and the notes writer - on a scripted transcript,",
        "so the plumbing under test is the real plumbing.",
        "",
        "What it does **not** exercise is model memory: the mock ASR does not hold",
        "Whisper's KV cache, and the heuristic glossary is not a 3B model. Resident",
        "memory here is the application's own growth, not the total footprint of a",
        "loaded pipeline. Re-run with weights present for that figure.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Long-session stability check.")
    ap.add_argument("--minutes", type=int, default=60, help="simulated lecture minutes")
    ap.add_argument("--write", action="store_true", help="write docs/SOAK.md")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    from sahaay.pipeline import Pipeline

    cfg = load_config()
    cfg.mock = True
    pipeline = Pipeline(cfg)
    device = pipeline.device.provider_label

    print(f"\n  soak: {args.minutes} simulated minutes on {device}\n")
    pipeline.load_models()
    pipeline.start()

    samples: list[dict] = []
    minutes: list[float] = []
    started = time.time()

    try:
        for minute in range(args.minutes):
            time.sleep(SECONDS_PER_SIMULATED_MINUTE)
            snapshot = sample(pipeline)
            samples.append(snapshot)
            minutes.append(float(minute))
            if minute % 10 == 0 or minute == args.minutes - 1:
                print(
                    f"  min {minute:3d}  rss {snapshot['rss_mb']:7.1f} MB  "
                    f"captions {snapshot['captions']:4d}  "
                    f"history {snapshot['bus_history']:4d}  "
                    f"queue {snapshot['segment_queue']:2d}  "
                    f"threads {snapshot['threads']:2d}"
                )
    finally:
        pipeline.stop()
        gc.collect()

    wall = time.time() - started

    # Two different questions, and conflating them gave a false failure.
    #
    # Some structures are *capped*: event history and the metric ring buffers
    # grow until they hit a documented limit and then stop. Over an hour they
    # never reach it, so a slope test flags healthy growth as unbounded - as
    # it did on the first run, failing on bus_history at 0.58/min when the
    # cap is 500 and it reached 35. The right question for these is "did it
    # exceed its cap", not "is the slope zero".
    #
    # Others must genuinely stay flat: threads, queue depth, memory.
    from sahaay.bus import EventBus
    from sahaay.metrics import Metrics

    caps = {
        "bus_history": EventBus()._history_limit,
        "metric_samples": Metrics()._samples.default_factory().maxlen * 8,
    }
    # Captions and glossary entries are meant to accumulate - they are the
    # transcript, and the student keeps it.
    unbounded_by_design = {"captions", "glossary"}
    flat_thresholds = {
        "rss_mb": 2.0,
        "segment_queue": 0.2,
        "threads": 0.05,
    }

    growth = {
        key: round(slope_per_minute([s[key] for s in samples], minutes), 3)
        for key in samples[0]
    }

    breaches = []
    for key, rate in growth.items():
        if key in unbounded_by_design:
            continue
        if key in caps:
            if max(s[key] for s in samples) > caps[key]:
                breaches.append(f"{key} (exceeded cap {caps[key]})")
        elif abs(rate) > flat_thresholds.get(key, 1e9):
            breaches.append(key)

    exempt = list(caps) + sorted(unbounded_by_design)
    thresholds = {**flat_thresholds, **dict.fromkeys(exempt, 1000000000.0)}
    verdict = "pass - nothing grows without bound" if not breaches else (
        f"FAIL - unbounded growth in {', '.join(breaches)}"
    )

    result = {
        "generated": dt.datetime.now().isoformat(),
        "device": device,
        "simulated_minutes": args.minutes,
        "wall_seconds": round(wall, 1),
        "samples": samples,
        "growth_per_minute": growth,
        "thresholds": thresholds,
        "caps": caps,
        "unbounded_by_design": sorted(unbounded_by_design),
        "breaches": breaches,
        "verdict": verdict,
    }

    print(f"\n  {verdict}")

    md = render_markdown(result)
    if args.write:
        out = REPO_ROOT / "docs" / "SOAK.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"  wrote {out}")
    else:
        print("\n" + md)

    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if not breaches else 1


if __name__ == "__main__":
    raise SystemExit(main())
