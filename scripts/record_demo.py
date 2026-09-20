"""Record a session's event stream so the hosted demo replays real data.

The public site at https://sahaay.vercel.app cannot run the pipeline - there
is no NPU in a datacenter, no microphone, and the models are gigabytes. What
it *can* do honestly is replay a session that did run, through the same UI,
driven by the same events the browser receives over the WebSocket.

So this drives the real :class:`~sahaay.pipeline.Pipeline` and dumps the real
:class:`~sahaay.bus.EventBus` history to JSON. Nothing in the recording is
written by hand: every caption, translation, glossary entry, metric and the
session notes come out of the pipeline exactly as the live UI would see them.

Timestamps are rewritten relative to the first event so the replay can honour
the original spacing without leaking the recording date into the payload.

    python scripts/record_demo.py                  # mock pipeline, no weights
    python scripts/record_demo.py --captions 12
    python scripts/record_demo.py --real           # with models present

``--mock`` (the default) uses the scripted transcript, which means the site
can be rebuilt on any machine with no model downloads. The recording it
produces is a faithful *event stream* but the text is scripted rather than
transcribed, and the demo page says so in as many words. Run ``--real`` on a
machine with weights to replace it with a genuine transcription.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahaay.config import load_config  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "web" / "session.json"


def _capture(bus) -> list:
    """Tap every published event, not just the ones kept in history.

    ``EventBus`` deliberately drops ``partial`` and ``metric`` from its
    replay history - they are transient, and re-sending them to a browser
    that connects late would show superseded text. But they are exactly
    what a *replay* needs: without the metric stream the RTF readout never
    updates, and the demo would be visibly less alive than the product.

    So this wraps publish rather than reading history afterwards, which
    captures the stream a live WebSocket subscriber would actually receive.
    """
    captured: list = []
    original = bus.publish

    def publish(kind, **data):
        event = original(kind, **data)
        captured.append(event)
        return event

    bus.publish = publish
    return captured


def _drained(pipeline, captured: list, timeout_s: float) -> bool:
    """Wait for the segment queue to empty and stay empty.

    "Empty" alone is not enough: the queue is also empty while the
    transcriber holds the last segment, so this waits for two consecutive
    quiet checks with no new event in between.
    """
    deadline = time.time() + timeout_s
    seen = len(captured)
    quiet = 0
    while time.time() < deadline:
        if pipeline._segments.empty() and len(captured) == seen:
            quiet += 1
            if quiet >= 2:
                return True
        else:
            quiet = 0
            seen = len(captured)
        time.sleep(1.0)
    return False


def redact(payload: dict) -> dict:
    """Strip anything machine-specific before the recording goes public.

    The notes event carries the absolute path the session was saved to,
    which on a development machine is a home directory. Publishing that to
    a website is a small leak, but it is a leak, and it is the kind that
    ships because nobody thought to look at the field.

    tests/test_web_build.py asserts the result, so this cannot regress.
    """
    home = str(Path.home())
    root = str(REPO_ROOT)

    def clean(value):
        if isinstance(value, str):
            for prefix in (root, home):
                value = value.replace(prefix + "\\", "").replace(prefix + "/", "")
            return value.replace(root, "").replace(home, "")
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    for event in payload["events"]:
        if event.get("kind") == "notes" and event.get("path"):
            # Keep the shape - the UI renders "Saved to ..." - but only the
            # part a reader learns anything from.
            event["path"] = "sessions/" + Path(event["path"]).name

    return clean(payload)


def record(
    captions: int,
    mock: bool,
    language: str,
    timeout_s: float,
    audio_file: Path | None,
    rate: float,
) -> dict:
    from sahaay.pipeline import Pipeline

    cfg = load_config()
    cfg.mock = mock
    cfg.translate.target_language = language
    cfg.audio_file = audio_file
    cfg.audio_rate = rate

    pipeline = Pipeline(cfg)
    captured = _capture(pipeline.bus)
    pipeline.load_models()
    pipeline.start()

    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            finalised = sum(1 for e in captured if e.kind == "caption")
            if finalised >= captions:
                break
            # The file ran out. Give the transcriber time to drain the queue -
            # on CPU a translation can take seconds, and cutting here would
            # truncate the recording mid-caption.
            exhausted = getattr(pipeline._source, "exhausted", False)
            if exhausted and _drained(pipeline, captured, timeout_s=60.0):
                break
            time.sleep(0.1)
    finally:
        # stop() runs the summariser, which publishes the notes event. The
        # sheet is part of the product, so it belongs in the recording.
        pipeline.stop()

    events = list(captured)
    if not events:
        raise SystemExit("recorded nothing - the pipeline published no events")

    origin = events[0].ts
    payload = []
    for e in events:
        payload.append({"kind": e.kind, "t": round(e.ts - origin, 3), **e.data})

    kinds: dict[str, int] = {}
    for e in payload:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1

    return redact({
        "recorded_with": "scripts/record_demo.py",
        "mode": "mock" if mock else "real",
        "audio_file": audio_file.name if audio_file else None,
        "target_language": language,
        "duration_s": round(events[-1].ts - origin, 3),
        "event_counts": kinds,
        "events": payload,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captions", type=int, default=10, help="stop after this many finalised captions")
    ap.add_argument("--language", default="hi", help="translation target language code")
    ap.add_argument("--timeout", type=float, default=120.0, help="give up after this many seconds")
    ap.add_argument("--real", action="store_true", help="use real models instead of the scripted transcript")
    ap.add_argument("--audio-file", type=Path, help="WAV to play through the pipeline")
    ap.add_argument("--audio-rate", type=float, default=1.0, help="play at X times real time")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.audio_file and not args.audio_file.exists():
        raise SystemExit(f"no such audio file: {args.audio_file}")

    data = record(
        args.captions,
        mock=not args.real,
        language=args.language,
        timeout_s=args.timeout,
        audio_file=args.audio_file,
        rate=args.audio_rate,
    )

    out = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    counts = ", ".join(f"{k} x{v}" for k, v in sorted(data["event_counts"].items()))
    try:
        shown = out.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"wrote {shown}")
    print(f"  mode      {data['mode']}")
    print(f"  duration  {data['duration_s']:.1f}s")
    print(f"  events    {len(data['events'])} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
