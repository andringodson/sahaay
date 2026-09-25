"""Does the browser build actually transcribe? Drive it with a fake microphone.

web/live/ runs Whisper in the visitor's browser. Nothing in the test suite
can tell you whether that works: the suite checks files, and this depends on
a CDN fetch, a WASM compile, a WebGPU probe, microphone permission and the
page's own content-security policy. Every one of those fails silently into
"press Start and nothing happens".

So this launches Chromium with its fake capture device fed from a real WAV,
presses Start, and waits for a caption to appear. Chromium plays the file
into getUserMedia as though someone were speaking it.

    python scripts/check_live.py
    python scripts/check_live.py --base https://sahaay-offline.vercel.app

The --base run is the one that matters. The content-security policy only
exists on the deployment, and a policy that blocks the runtime produces a
page that loads perfectly and never captions - which is exactly how the CSP
caught out scripts/shoot_web.py once already.

The first run downloads the model, so allow a few minutes.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import socket
import threading
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
DEFAULT_WAV = REPO_ROOT / "testaudio" / "lecture_long.wav"


def free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(directory: Path):
    port = free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None  # type: ignore[method-assign]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def check_wav(path: Path) -> None:
    """Chromium's fake device wants 16-bit PCM or it feeds silence."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path.name} is not 16-bit PCM; Chromium would play silence")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="check a deployment instead of the local web/")
    ap.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="audio to speak into the page")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for a caption")
    args = ap.parse_args()

    if not args.wav.exists():
        raise SystemExit(f"no such audio: {args.wav}\n  python scripts/make_lecture_audio.py")
    check_wav(args.wav)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n  pip install playwright\n  playwright install chromium")
        return 1

    problems: list[str] = []
    source = contextlib.nullcontext(args.base.rstrip("/")) if args.base else serve(WEB_DIR)

    with source as base, sync_playwright() as pw:
        print(f"  checking {base}/live/")
        print(f"  speaking {args.wav.name} into the page\n")

        browser = pw.chromium.launch(
            headless=not args.headed,
            args=[
                "--use-fake-ui-for-media-stream",      # grant the mic without a prompt
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={args.wav}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        page.goto(f"{base}/live/", wait_until="networkidle")
        page.wait_for_selector("#toggle:not([disabled])", timeout=30_000)
        page.click("#toggle")

        try:
            # Generous: the first press fetches the model over the network.
            page.wait_for_selector("#captions li", timeout=args.timeout * 1000)
            page.wait_for_selector(
                "#captions li:nth-child(2)", timeout=max(60, args.timeout // 2) * 1000
            )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"no captions appeared: {exc}")

        captions = page.locator("#captions li").all_inner_texts()
        badge = page.inner_text("#device-label")
        glossary = page.locator("#glossary li").count()
        status_line = page.inner_text("#live-status") if page.locator("#live-status").count() else ""

        browser.close()

    print(f"  backend    {badge}")
    print(f"  status     {status_line.strip() or '-'}")
    print(f"  captions   {len(captions)}")
    for line in captions[:4]:
        print(f"    {line.splitlines()[0][:90]}")
    print(f"  glossary   {glossary}")

    # A CSP violation reads as an ordinary console error, and it is the one
    # failure that only ever shows up on the deployment.
    blocked = [c for c in console if "Content Security Policy" in c or "Refused to" in c]
    if blocked:
        problems.append("content-security-policy blocked something:")
        problems.extend(f"    {c}" for c in blocked[:4])
    if not captions:
        problems.append("no captions at all - inference never produced text")

    if problems:
        print("\nFAILED")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\n  OK - Whisper ran in the browser and captioned real audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
