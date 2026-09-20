"""Command line entry point: ``python -m sahaay``.

Three modes, because they serve three different audiences:

* no flags      - launch the UI (what a student uses)
* ``--cli``     - captions in the terminal (what you debug with)
* ``--device``  - print the execution provider and exit (what a judge runs
                  first to check the NPU claim is real)
* ``--selftest`` - check every dependency and report ok/degraded/failed
                   (what a judge runs second, when something looks wrong)
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import time

from .config import SUPPORTED_LANGUAGES, load_config


def _force_utf8_console() -> None:
    """Make stdout survive Devanagari, Tamil and Malayalam.

    A Windows console defaults to cp1252, and printing a Hindi translation
    to it raises UnicodeEncodeError - which would crash --cli mode the
    moment the first translated caption arrived. Reconfiguring is safe:
    anything that cannot be represented is replaced rather than fatal.
    """
    for stream in (sys.stdout, sys.stderr):
        # AttributeError: not a real stream (pytest capture, a pipe).
        # ValueError: already detached.
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    )
    # uvicorn's access log would interleave with captions in --cli mode.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _print_device() -> int:
    from .runtime import SessionFactory

    try:
        # Use the real config so SAHAAY_PROVIDER is honoured here too -
        # otherwise `--device` reports something the app would not do.
        report = SessionFactory(load_config().runtime).report()
    except RuntimeError as exc:
        print(exc)
        return 1

    print()
    print("  Sahaay device report")
    print("  " + "-" * 46)
    print(f"  Execution provider : {report.provider_label}")
    print(f"  ONNX Runtime       : {report.ort_version}")
    print(f"  Architecture       : {report.machine}{' (ARM64)' if report.is_arm64 else ''}")
    print(f"  Processor          : {report.processor}")
    print(f"  Available providers: {', '.join(report.available_providers) or 'none'}")
    print(f"  Hexagon NPU active : {'yes' if report.npu_active else 'no'}")
    if report.fallback_reason:
        print(f"  Fallback reason    : {report.fallback_reason}")
    print()
    return 0


def _run_cli(cfg) -> int:
    """Terminal captions. Useful when the browser is in the way of a demo."""
    from . import bus as ev
    from .bus import EventBus
    from .pipeline import Pipeline

    bus = EventBus()
    pipeline = Pipeline(cfg, bus)

    print(f"\n  device: {pipeline.device.summary()}\n  loading models...\n")
    pipeline.load_models()

    # Without an asyncio loop the bus only stores history, so subscribe by
    # monkey-patching publish: simplest thing that keeps the bus honest.
    original_publish = bus.publish

    def echo(kind: str, **data):
        if kind == ev.CAPTION:
            print(f"  [{data.get('start_s', 0):6.1f}s] {data.get('text', '')}")
        elif kind == ev.TRANSLATION and not data.get("passthrough"):
            print(f"             → {data.get('text', '')}")
        elif kind == ev.GLOSS:
            print(f"      * {data.get('term')}: {data.get('explanation')}")
        elif kind == ev.ERROR:
            print(f"  !! {data.get('message')}")
        return original_publish(kind, **data)

    bus.publish = echo  # type: ignore[method-assign]

    stopping = False

    def handle_sigint(signum, frame):  # noqa: ANN001, ARG001
        nonlocal stopping
        if stopping:
            sys.exit(1)
        stopping = True
        print("\n  stopping; writing notes...\n")

    signal.signal(signal.SIGINT, handle_sigint)

    pipeline.start()
    print("  listening. Ctrl-C to stop and save notes.\n")
    try:
        while not stopping:
            time.sleep(0.2)
    finally:
        notes = pipeline.stop()
        if notes:
            print(f"\n  saved {len(notes.captions)} captions, {len(notes.glossary)} terms")
        else:
            print("\n  nothing captured")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sahaay",
        description="Offline live lecture captioning for Snapdragon-powered PCs.",
    )
    parser.add_argument("--cli", action="store_true", help="print captions in the terminal")
    parser.add_argument("--device", action="store_true", help="show the execution provider and exit")
    parser.add_argument(
        "--selftest", action="store_true",
        help="check everything and report what works, then exit",
    )
    parser.add_argument("--quick", action="store_true", help="with --selftest, skip the pipeline run")
    parser.add_argument(
        "--mock", action="store_true", help="run without models, using a scripted transcript"
    )
    parser.add_argument(
        "--lang", choices=sorted(SUPPORTED_LANGUAGES), help="translate captions into this language"
    )
    parser.add_argument("--port", type=int, help="UI port (default 8756)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _force_utf8_console()
    _setup_logging(args.verbose)

    if args.device:
        return _print_device()

    cfg = load_config()

    if args.selftest:
        from .selftest import report, run

        checks, code = run(cfg, quick=args.quick)
        print(report(checks))
        return code
    if args.mock:
        cfg.mock = True
    if args.lang:
        cfg.translate.target_language = args.lang
    if args.port:
        cfg.server.port = args.port
    if args.no_browser:
        cfg.server.open_browser = False

    if args.cli:
        return _run_cli(cfg)

    from .server import serve

    serve(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
