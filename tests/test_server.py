"""HTTP and WebSocket surface.

The WebSocket tests exist because of a specific bug that cost real time and
produced no error message: with ``from __future__ import annotations`` in
server.py, FastAPI resolved the ``websocket: WebSocket`` annotation as the
string "WebSocket" against module globals, failed, and closed every
connection on accept. The page loaded fine and simply never updated.

``test_websocket_accepts_a_connection`` is the regression guard.
"""

import json
import time

import pytest

from sahaay.config import load_config

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

from sahaay.server import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    cfg = load_config()
    cfg.mock = True
    cfg.server.open_browser = False
    cfg.sessions_dir = tmp_path / "sessions"
    cfg.models_dir = tmp_path / "models"
    with TestClient(create_app(cfg)) as c:
        yield c


class TestPages:
    def test_index_is_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    @pytest.mark.parametrize("path", ["/static/app.js", "/static/style.css", "/static/index.html"])
    def test_static_assets(self, client, path):
        assert client.get(path).status_code == 200


class TestApi:
    def test_status_shape(self, client):
        d = client.get("/api/status").json()
        assert "device" in d and "provider_label" in d["device"]
        assert isinstance(d["running"], bool)
        assert len(d["languages"]) >= 8

    def test_status_is_json_serialisable(self, client):
        json.dumps(client.get("/api/status").json())

    def test_language_switch(self, client):
        assert client.post("/api/language/ta").json()["target_language"] == "ta"

    def test_unknown_language_is_rejected(self, client):
        assert client.post("/api/language/zz").status_code == 400

    def test_sessions_listing(self, client):
        assert "sessions" in client.get("/api/sessions").json()

    def test_stop_without_start_is_safe(self, client):
        assert client.post("/api/stop").json()["ok"] is True


class TestWebSocket:
    def test_websocket_accepts_a_connection(self, client):
        # The regression guard. This failed silently for every connection
        # until the annotation-resolution bug in server.py was found.
        with client.websocket_connect("/ws") as ws:
            first = json.loads(ws.receive_text())
            assert first["kind"] == "status"

    def test_first_message_carries_the_device(self, client):
        # A page connecting mid-lecture must render the EP badge immediately
        # rather than waiting for the next event.
        with client.websocket_connect("/ws") as ws:
            first = json.loads(ws.receive_text())
            assert "device" in first
            assert "npu_active" in first["device"]

    def test_full_event_flow(self, client):
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())  # initial status
            assert client.post("/api/start").json()["ok"] is True

            kinds: dict[str, int] = {}
            deadline = time.time() + 25
            while time.time() < deadline:
                msg = json.loads(ws.receive_text())
                kinds[msg["kind"]] = kinds.get(msg["kind"], 0) + 1
                if kinds.get("caption", 0) >= 2 and kinds.get("gloss", 0) >= 1:
                    break

            assert kinds.get("caption", 0) >= 2, f"no captions arrived: {kinds}"
            assert kinds.get("translation", 0) >= 1
            assert kinds.get("gloss", 0) >= 1

        assert client.post("/api/stop").json()["saved"] is True

    def test_caption_payload_has_what_the_ui_needs(self, client):
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())
            client.post("/api/start")

            deadline = time.time() + 25
            caption = None
            while time.time() < deadline and caption is None:
                msg = json.loads(ws.receive_text())
                if msg["kind"] == "caption":
                    caption = msg

            assert caption is not None
            for key in ("index", "text", "start_s", "latency_ms", "rtf", "provider"):
                assert key in caption, f"caption missing {key}"

        client.post("/api/stop")
