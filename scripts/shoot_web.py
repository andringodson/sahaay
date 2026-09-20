"""Render the public site in a real browser, then look at it.

The last round of UI bugs in this repo were both invisible to the test suite
and obvious the moment anyone rendered the page: a notes panel that covered
the captions because a class rule beat ``[hidden]``, and a WebSocket endpoint
that 404'd while the page loaded perfectly. Structural tests cannot see
either. A browser can.

So this serves web/ over real HTTP, drives the demo the way a visitor would -
press Start, wait for captions - and fails if the page is not actually
working. The screenshots are a by-product; the assertions are the point.

    python scripts/shoot_web.py
    python scripts/shoot_web.py --headed      # watch it happen
    python scripts/shoot_web.py --base https://sahaay-offline.vercel.app

Requires playwright (``pip install playwright && playwright install chromium``).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import socket
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
OUT_DIR = REPO_ROOT / "docs" / "img"


def shown(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    --out can point anywhere, and relative_to raises rather than falling
    back when it cannot.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(directory: Path):
    """A static server, because file:// breaks fetch() and relative paths."""
    port = free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None  # type: ignore[method-assign]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--lang", default="hi", help="which recording to drive")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--base", metavar="URL",
        help="check a deployed site instead of the local web/ directory",
    )
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n  pip install playwright\n  playwright install chromium")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    # A local copy passing is not the same as the deployment passing - the
    # headers, the rewrite rules and the CSP only exist on the real host.
    source = contextlib.nullcontext(args.base.rstrip("/")) if args.base else serve(WEB_DIR)

    with source as base, sync_playwright() as pw:
        print(f"checking {base}")
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)

        # A page that logs errors is a page that is broken, even if it looks
        # fine in a screenshot.
        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: console.append(f"requestfailed: {r.url}"))

        # -- landing page --------------------------------------------------
        page.goto(f"{base}/", wait_until="networkidle")
        if not args.base:
            page.screenshot(path=str(args.out / "site.png"), full_page=True)
            print(f"wrote {shown(args.out / 'site.png')}")

        if not page.locator("h1").first.is_visible():
            problems.append("landing page has no visible heading")

        # -- demo ----------------------------------------------------------
        page.goto(f"{base}/demo/?lang={args.lang}&play=1&speed=4", wait_until="networkidle")

        try:
            # Selector waits, not wait_for_function: Playwright evaluates a
            # string predicate with eval(), and the deployed CSP forbids
            # unsafe-eval. The check has to work against the real headers or
            # it is only ever testing the local copy.
            page.wait_for_selector("#captions li", timeout=30_000)
            page.wait_for_selector("#captions li:nth-child(6)", timeout=60_000)
            # The jargon sidebar is half the product's pitch. Waiting for it
            # means the screenshot shows it and a silent glossary fails here.
            page.wait_for_selector("#glossary li", timeout=60_000)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"replay did not reach a usable state: {exc}")

        # Mid-session the button has to say Stop. It said Start, because the
        # recording replays the pipeline's own lifecycle events.
        toggle = page.locator("#toggle").inner_text().strip()
        if toggle.lower() not in {"stop", "saving…", "saving..."}:
            problems.append(f"Start button reads {toggle!r} while the replay is running")

        captions = page.locator("#captions li").count()
        glossary = page.locator("#glossary li").count()
        badge = page.locator("#device-label").inner_text()
        rtf = page.locator("#rtf").inner_text()

        # The notes sheet must stay hidden until the session ends. It sat
        # permanently on top of the captions once, and only a screenshot
        # showed it.
        notes_visible = page.locator("#notes").is_visible()
        if notes_visible:
            problems.append("the session-notes sheet is covering the captions")

        if not args.base:
            page.screenshot(path=str(args.out / "demo.png"))
            print(f"wrote {shown(args.out / 'demo.png')}")

        print(f"\n  captions   {captions}")
        print(f"  glossary   {glossary}")
        print(f"  device     {badge}")
        print(f"  rtf        {rtf}")

        if captions == 0:
            problems.append("no captions rendered")
        if "connecting" in badge.lower():
            problems.append(f"device badge stuck on {badge!r} - status never arrived")

        # -- phone width ---------------------------------------------------
        # The demo banner is extra vertical chrome the product does not have,
        # and body has overflow:hidden. On a narrow screen that is exactly
        # where the caption area gets squeezed to nothing.
        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        phone.goto(f"{base}/demo/?lang={args.lang}&play=1&speed=8", wait_until="networkidle")
        try:
            phone.wait_for_selector("#captions li", timeout=30_000)
        except Exception:  # noqa: BLE001
            problems.append("no captions at phone width")

        caption_box = phone.locator("#captions").bounding_box()
        if caption_box and caption_box["height"] < 150:
            problems.append(
                f"caption area is only {caption_box['height']:.0f}px tall at 390px wide"
            )

        if not args.base:
            phone.screenshot(path=str(args.out / "demo-mobile.png"))
            print(f"wrote {shown(args.out / 'demo-mobile.png')}")
        if caption_box:
            print(f"  caption area at 390px: {caption_box['height']:.0f}px tall")

        phone.close()
        browser.close()

    if console:
        problems.extend(console)

    if problems:
        print("\nFAILED")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nOK - the site renders and the demo replays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
