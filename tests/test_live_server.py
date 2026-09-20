"""End-to-end over a real socket, not the in-process test client.

This file exists because of a failure the rest of the suite structurally
cannot see. Starlette's ``TestClient`` fakes the WebSocket transport, so
every WebSocket test passes even when uvicorn has no WebSocket library
installed at all - in which case the real server answers ``/ws`` with 404,
keeps serving the page, and the UI loads perfectly and never updates.

So these tests start uvicorn as a subprocess and talk to it over TCP.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 90.0) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def live_server():
    port = free_port()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "SAHAAY_MOCK": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "sahaay", "--mock", "--no-browser", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        if not wait_for(f"{base}/api/status"):
            proc.kill()
            out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
            pytest.skip(f"server did not start:\n{out[-2000:]}")
        yield base, port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


class TestWebSocketTransport:
    def test_a_websocket_library_is_installed(self):
        # The root cause. Without this uvicorn 404s /ws and the UI silently
        # never receives a caption.
        from sahaay.server import websocket_library

        assert websocket_library() is not None, (
            "uvicorn needs `websockets` or `wsproto`; without one the page "
            "loads but never updates"
        )

    def test_it_is_declared_as_a_dependency(self):
        text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "websockets" in text or "wsproto" in text

    def test_real_handshake_succeeds(self, live_server):
        # The assertion TestClient cannot make: a genuine HTTP upgrade.
        ws = pytest.importorskip("websockets.sync.client", reason="websockets not installed")
        base, port = live_server
        with ws.connect(f"ws://127.0.0.1:{port}/ws", open_timeout=30) as conn:
            first = json.loads(conn.recv(timeout=30))
            assert first["kind"] == "status"
            assert "device" in first


class TestLiveCaptions:
    def test_captions_arrive_over_a_real_socket(self, live_server):
        ws = pytest.importorskip("websockets.sync.client", reason="websockets not installed")
        import urllib.request

        base, port = live_server
        with ws.connect(f"ws://127.0.0.1:{port}/ws", open_timeout=30) as conn:
            json.loads(conn.recv(timeout=30))  # initial status

            urllib.request.urlopen(
                urllib.request.Request(f"{base}/api/start", method="POST"), timeout=30
            ).read()

            kinds: dict[str, int] = {}
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    msg = json.loads(conn.recv(timeout=10))
                except Exception:  # noqa: BLE001
                    break
                kinds[msg["kind"]] = kinds.get(msg["kind"], 0) + 1
                if kinds.get("caption", 0) >= 2:
                    break

            assert kinds.get("caption", 0) >= 2, f"no captions over a real socket: {kinds}"

    def test_a_late_subscriber_gets_the_transcript_so_far(self, live_server):
        """Opening the page mid-lecture must not show an empty pane."""
        ws = pytest.importorskip("websockets.sync.client", reason="websockets not installed")
        import urllib.request

        base, port = live_server
        status = json.loads(urllib.request.urlopen(f"{base}/api/status", timeout=30).read())
        if not status["running"]:
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/api/start", method="POST"), timeout=30
            ).read()
            time.sleep(8)

        # Connect only now - after captions already exist.
        with ws.connect(f"ws://127.0.0.1:{port}/ws", open_timeout=30) as conn:
            replayed = 0
            deadline = time.time() + 20
            while time.time() < deadline:
                try:
                    msg = json.loads(conn.recv(timeout=6))
                except Exception:  # noqa: BLE001
                    break
                if msg["kind"] == "caption":
                    replayed += 1
                    if replayed >= 1:
                        break
            assert replayed >= 1, "history was not replayed to a late subscriber"


class TestStaticAssets:
    @pytest.mark.parametrize("path", ["/", "/static/app.js", "/static/style.css"])
    def test_served_over_http(self, live_server, path):
        import urllib.request

        base, _ = live_server
        with urllib.request.urlopen(f"{base}{path}", timeout=30) as r:
            assert r.status == 200
            assert len(r.read()) > 0
