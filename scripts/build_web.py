"""Assemble web/ - the public site - from the product's own UI files.

The demo page must not be a reimplementation of the UI. If it were, it would
drift: someone fixes a rendering bug in ``sahaay/ui/app.js``, the hosted demo
keeps the bug, and the site quietly stops representing the product.

So the demo *is* the product's UI. ``app.js`` and ``style.css`` are copied
byte for byte, ``index.html`` is rewritten only for asset paths, and the one
new file - ``replay.js`` - swaps the transport underneath: it stubs
``fetch`` and ``WebSocket`` so the same rendering code is driven by a
recorded event stream instead of a live pipeline.

``tests/test_web_build.py`` fails if the copies drift, so forgetting to
re-run this is a CI failure rather than a stale website.

    python scripts/build_web.py
    python scripts/build_web.py --check     # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

UI_DIR = REPO_ROOT / "sahaay" / "ui"
WEB_DIR = REPO_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
DEMO_DIR = WEB_DIR / "demo"

# Copied verbatim. Anything that needs changing for the web is changed by
# replay.js at runtime, not by editing these.
VERBATIM = ["app.js", "style.css"]

BANNER = """
<div class="replay-banner">
  <strong>Recorded session.</strong>
  The real interface, the real rendering code, replaying events a real run
  published. <span class="replay-long">Nothing is inferring on this page &mdash;
  that happens on your machine, which is the entire point.
  <span id="replay-origin"></span></span>
  <a href="../">What ran on Snapdragon &rarr;</a>
</div>
""".strip()

BANNER_CSS = """
/* Added by scripts/build_web.py - not part of the product UI. */
.replay-banner {
  padding: 0.6rem var(--gap);
  background: var(--accent-dim);
  color: var(--text);
  font-size: 0.85rem;
  line-height: 1.5;
  border-bottom: 1px solid var(--line);
}
.replay-banner a { color: var(--accent); white-space: nowrap; }
.replay-controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-right: 0.25rem;
  font-size: 0.8rem;
  color: var(--text-dim);
}

/* On a phone the banner is pure overhead above the thing people came to
   see. body is overflow:hidden, so every line it takes comes straight out
   of the caption area - measured at 144px of captions on a 844px screen.
   Drop the explanation and the replay-speed control; the link stays. */
@media (max-width: 40rem) {
  .replay-banner { font-size: 0.78rem; padding: 0.45rem var(--gap); }
  .replay-long { display: none; }
  .replay-controls { display: none; }
}
""".strip()


def build_demo_html(source: str) -> str:
    """Rewrite the product's index.html for static hosting."""
    out = source
    # The server mounts the UI at /static; on the site it sits one level up.
    out = out.replace('href="/static/', 'href="../static/')
    out = out.replace('src="/static/', 'src="../static/')

    out = out.replace(
        "<title>Sahaay</title>",
        "<title>Sahaay — recorded session</title>",
    )

    # replay.js must be evaluated before app.js: it installs the fetch and
    # WebSocket stubs that app.js reaches for the moment it boots.
    out = out.replace(
        '<script src="../static/app.js"></script>',
        '<script src="../static/replay.js"></script>\n'
        '<script src="../static/app.js"></script>',
    )

    out = out.replace("<body>", "<body>\n\n" + BANNER, 1)

    # A stylesheet of its own rather than appending to the product's, so the
    # verbatim copy stays verbatim and the drift test stays meaningful.
    out = out.replace(
        '<link rel="stylesheet" href="../static/style.css">',
        '<link rel="stylesheet" href="../static/style.css">\n'
        '<link rel="stylesheet" href="../static/replay.css">',
    )
    return out


def expected_files() -> dict[Path, str]:
    """Every generated path mapped to the content it should hold."""
    files: dict[Path, str] = {}
    for name in VERBATIM:
        files[STATIC_DIR / name] = (UI_DIR / name).read_text(encoding="utf-8")
    files[DEMO_DIR / "index.html"] = build_demo_html(
        (UI_DIR / "index.html").read_text(encoding="utf-8")
    )
    files[STATIC_DIR / "replay.css"] = BANNER_CSS + "\n"
    return files


def sessions_index() -> dict:
    """List the recordings that shipped, for the language picker."""
    from sahaay.config import SUPPORTED_LANGUAGES

    # Ordered as SUPPORTED_LANGUAGES is, not alphabetically: the demo opens
    # on the first entry, and Hindi is the one to open on.
    order = list(SUPPORTED_LANGUAGES)
    paths = sorted(
        WEB_DIR.glob("session.*.json"),
        key=lambda p: (order.index(p.name.split(".")[1])
                       if p.name.split(".")[1] in order else len(order), p.name),
    )

    found = []
    for path in paths:
        code = path.name.split(".")[1]
        meta = json.loads(path.read_text(encoding="utf-8"))
        found.append(
            {
                "code": code,
                "name": SUPPORTED_LANGUAGES.get(code, {}).get("name", code),
                "file": path.name,
                "mode": meta.get("mode", "unknown"),
                "captions": meta.get("event_counts", {}).get("caption", 0),
                "duration_s": meta.get("duration_s"),
            }
        )
    return {"sessions": found}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()

    files = expected_files()

    stale = [p for p, content in files.items() if not p.exists() or p.read_text(encoding="utf-8") != content]

    if args.check:
        if stale:
            for p in stale:
                print(f"stale: {p.relative_to(REPO_ROOT)}")
            print("\nrun: python scripts/build_web.py")
            return 1
        print("web/ is up to date with sahaay/ui/")
        return 0

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    index = sessions_index()
    (WEB_DIR / "sessions.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    for path in sorted(files):
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    langs = ", ".join(s["code"] for s in index["sessions"]) or "none"
    print(f"wrote web/sessions.json ({len(index['sessions'])} recordings: {langs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
