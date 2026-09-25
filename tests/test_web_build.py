"""The hosted site must keep representing the product.

web/ is a copy of the product UI plus a replay harness. Copies rot: someone
fixes a rendering bug in sahaay/ui/app.js, nobody re-runs the build, and the
public demo keeps the bug while claiming to be the application. These tests
make that a CI failure instead of a stale website.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

WEB = REPO_ROOT / "web"
UI = REPO_ROOT / "sahaay" / "ui"


def test_web_build_is_up_to_date():
    """web/ matches what scripts/build_web.py would generate right now."""
    import build_web

    stale = [
        path.relative_to(REPO_ROOT).as_posix()
        for path, content in build_web.expected_files().items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    assert not stale, (
        "web/ is out of date with sahaay/ui/: " + ", ".join(stale) +
        "\nrun: python scripts/build_web.py"
    )


def test_demo_uses_the_products_own_javascript():
    """Not a reimplementation - the same file, byte for byte."""
    assert (WEB / "static" / "app.js").read_bytes() == (UI / "app.js").read_bytes()
    assert (WEB / "static" / "style.css").read_bytes() == (UI / "style.css").read_bytes()


def test_replay_shim_loads_before_the_app():
    """app.js reaches for fetch and WebSocket as soon as it is evaluated.

    If replay.js were second, boot() would fire a real request at /api/status
    on a static host, get a 404, and the page would show an error toast.
    """
    html = (WEB / "demo" / "index.html").read_text(encoding="utf-8")
    assert html.index("replay.js") < html.index("app.js")


def test_demo_page_has_no_absolute_asset_paths():
    """The server mounts the UI at /static; the site nests it one level down."""
    html = (WEB / "demo" / "index.html").read_text(encoding="utf-8")
    assert '"/static/' not in html


def test_every_listed_recording_exists_and_parses():
    index = json.loads((WEB / "sessions.json").read_text(encoding="utf-8"))
    assert index["sessions"], "the demo ships no recordings; run scripts/record_demo.py"

    for entry in index["sessions"]:
        path = WEB / entry["file"]
        assert path.exists(), f"sessions.json lists {entry['file']}, which is missing"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["events"], f"{entry['file']} contains no events"


@pytest.mark.parametrize("path", sorted(WEB.glob("session.*.json")), ids=lambda p: p.name)
def test_recording_is_a_plausible_session(path: Path):
    """A recording has to be a session, not just well-formed JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    kinds = {e["kind"] for e in data["events"]}

    assert "caption" in kinds, "no captions - nothing to replay"
    assert "status" in kinds, "no status event - the device badge would stay blank"

    # Timestamps drive the replay. Out of order, captions would arrive
    # backwards; all-zero, the whole lecture would land in one frame.
    times = [e["t"] for e in data["events"]]
    assert times == sorted(times), "events are not in chronological order"
    assert times[-1] > 0, "every event carries the same timestamp"

    for event in data["events"]:
        if event["kind"] == "caption":
            assert event.get("text", "").strip(), "empty caption text"
            assert "index" in event, "caption without an index cannot be matched"


@pytest.mark.parametrize("path", sorted(WEB.glob("session.*.json")), ids=lambda p: p.name)
def test_recording_carries_no_local_paths(path: Path):
    """Nothing from the recording machine should ship to a public site.

    The notes event includes the path a session was saved to, which on a
    development machine is a home directory.
    """
    raw = path.read_text(encoding="utf-8")
    for leak in ("C:\\\\Users", "/home/", "/Users/"):
        assert leak not in raw, f"{path.name} leaks a local path ({leak})"


class TestLiveBuild:
    """The browser build is the product's UI with inference underneath.

    It has the same drift risk as the replay page and one extra: the seeded
    glossary is exported to JSON, and a change to SEED_GLOSSARY that is not
    re-exported leaves the site explaining a vocabulary the app no longer has.
    """

    def test_the_live_page_exists_and_loads_its_own_transport(self):
        html = (WEB / "live" / "index.html").read_text(encoding="utf-8")
        assert "live.js" in html
        assert html.index("live.js") < html.index("app.js"), (
            "live.js installs the fetch and WebSocket stubs app.js reaches for "
            "on boot; loading it second means app.js talks to a server that is "
            "not there."
        )

    def test_the_live_page_is_the_products_own_ui(self):
        live = (WEB / "live" / "index.html").read_text(encoding="utf-8")
        for element in ('id="captions"', 'id="glossary"', 'id="toggle"', 'id="level"'):
            assert element in live, f"{element} missing - the UI was reimplemented"

    def test_the_exported_glossary_matches_the_python_one(self):
        from sahaay.llm import SEED_GLOSSARY

        shipped = json.loads((WEB / "static" / "glossary.json").read_text(encoding="utf-8"))
        assert shipped == SEED_GLOSSARY, (
            "web/static/glossary.json is stale; run python scripts/build_web.py"
        )

    def test_the_live_page_says_what_it_cannot_do(self):
        """It runs Whisper, not the pipeline. The page has to say so."""
        landing = (WEB / "index.html").read_text(encoding="utf-8")
        assert "not" in landing and "NPU" in landing
        assert "translate" in landing.lower() or "translation" in landing.lower()

    def test_live_js_pins_its_runtime(self):
        """A floating major version can break a page mid-presentation."""
        js = (WEB / "static" / "live.js").read_text(encoding="utf-8")
        assert re.search(r"transformers@\d+\.\d+\.\d+", js), (
            "pin the transformers.js version rather than tracking latest"
        )

    def test_live_js_never_uploads_audio(self):
        """The whole claim. Nothing may POST audio anywhere."""
        js = (WEB / "static" / "live.js").read_text(encoding="utf-8")
        for pattern in ("FormData", "uploadAudio", "audio/wav"):
            assert pattern not in js, f"live.js references {pattern}"
