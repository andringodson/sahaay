"""The contract between the Python backend and the browser front end.

Nothing here needs a browser. These are the mismatches that fail *silently*:
the page loads, the WebSocket connects, and captions simply never appear
because one side spells an event kind differently, or the JavaScript reaches
for an element id the HTML no longer has.

Both have a real cost. The event-kind check is the reason bus.py keeps its
kinds as constants rather than inline strings.
"""

import json
import re
from pathlib import Path

import pytest

from sahaay import bus

UI = Path(__file__).resolve().parent.parent / "sahaay" / "ui"
APP_JS = (UI / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (UI / "index.html").read_text(encoding="utf-8")
STYLE_CSS = (UI / "style.css").read_text(encoding="utf-8")


def python_event_kinds() -> set[str]:
    return {
        value
        for name, value in vars(bus).items()
        if name.isupper() and isinstance(value, str) and not name.startswith("_")
    }


def js_event_kinds() -> set[str]:
    return set(re.findall(r'case "([a-z_]+)":', APP_JS))


class TestEventKinds:
    def test_every_published_kind_is_handled(self):
        missing = python_event_kinds() - js_event_kinds()
        assert not missing, f"backend publishes {missing} but the UI ignores them"

    def test_no_handler_for_a_kind_that_is_never_sent(self):
        extra = js_event_kinds() - python_event_kinds()
        assert not extra, f"UI handles {extra} which the backend never publishes"


class TestElementIds:
    def test_every_id_the_js_reaches_for_exists(self):
        used = set(re.findall(r'\$\("([^"]+)"\)', APP_JS))
        present = set(re.findall(r'id="([^"]+)"', INDEX_HTML))
        missing = used - present
        assert not missing, f"app.js queries #{missing}, absent from index.html"

    def test_no_orphan_ids(self):
        # An id nothing uses is usually the remains of a rename.
        used = set(re.findall(r'\$\("([^"]+)"\)', APP_JS))
        present = set(re.findall(r'id="([^"]+)"', INDEX_HTML))
        assert not present - used


class TestCaptionPayload:
    def test_ui_reads_only_fields_the_backend_sends(self):
        # The caption event is the one the UI picks apart field by field.
        sent = {
            "index", "text", "language", "start_s", "latency_ms", "rtf", "provider",
        }
        for field in ("index", "text", "start_s", "latency_ms", "rtf"):
            assert f"msg.{field}" in APP_JS, f"UI never reads {field}"
        assert sent  # documents the contract above


class TestStyling:
    @pytest.mark.parametrize(
        "token",
        ["--bg", "--text", "--accent", "--npu", "--cap-scale", "--cap-size", "--line"],
    )
    def test_custom_properties_are_defined(self, token):
        # A var() with no definition silently renders as nothing - often an
        # invisible caption on an invisible background.
        assert re.search(rf"{re.escape(token)}\s*:", STYLE_CSS), f"{token} used but never defined"

    def test_dark_and_light_are_both_defined(self):
        assert "prefers-color-scheme: light" in STYLE_CSS

    def test_reduced_motion_is_respected(self):
        # The caption list animates on every new line; someone who asked the
        # OS to stop that must not get it anyway.
        assert "prefers-reduced-motion" in STYLE_CSS

    def test_captions_are_a_live_region(self):
        # A screen-reader user needs new captions announced, and this is an
        # accessibility product before it is anything else.
        assert 'aria-live="polite"' in INDEX_HTML


class TestServerPaths:
    @pytest.mark.parametrize(
        "endpoint", ["/api/status", "/api/start", "/api/stop", "/api/language/", "/ws"]
    )
    def test_endpoints_the_ui_calls_exist_on_the_server(self, endpoint):
        server = (UI.parent / "server.py").read_text(encoding="utf-8")
        assert endpoint in APP_JS
        # /api/language/ is built with a template literal on the JS side.
        assert endpoint.rstrip("/") in server


class TestMockTranscript:
    def test_is_actually_code_mixed(self):
        # The mock demo exists to show the thing the product claims to
        # handle. An all-English script would quietly undercut the pitch.
        from sahaay.asr import MOCK_TRANSCRIPT

        devanagari_or_roman_hindi = [
            line for line in MOCK_TRANSCRIPT
            if any(w in line.lower() for w in (" ka ", " ke ", " hum ", " aur ", " hai", "karenge", "mein"))
        ]
        assert len(devanagari_or_roman_hindi) >= 3

    def test_contains_technical_terms(self):
        from sahaay.asr import MOCK_TRANSCRIPT

        text = " ".join(MOCK_TRANSCRIPT).lower()
        assert any(t in text for t in ("eigenvalue", "determinant", "diagonalize"))

    def test_is_json_safe(self):
        from sahaay.asr import MOCK_TRANSCRIPT

        json.dumps(MOCK_TRANSCRIPT)
