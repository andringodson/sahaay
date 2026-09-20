"""Local web UI: FastAPI + a WebSocket feed.

Bound to 127.0.0.1 only. A privacy-first product that opens a port on the
LAN would be contradicting itself, and the caption stream is exactly the
content the user was promised never leaves the machine.

A browser page rather than Electron or Tauri: it keeps the install to
``pip install -r requirements.txt``, which matters far more on Windows
ARM64, where prebuilt native wheels are the whole ballgame.
"""

#
# NOTE: this module deliberately does *not* use `from __future__ import
# annotations`. FastAPI resolves endpoint annotations with get_type_hints(),
# which looks them up in module globals. With postponed evaluation, the
# `websocket: WebSocket` annotation becomes the string "WebSocket", which
# cannot be resolved because fastapi is imported inside create_app() - and
# the route then silently closes every connection instead of accepting it.

import asyncio
import contextlib
import json
import logging
import webbrowser
from pathlib import Path

from .bus import EventBus
from .config import SUPPORTED_LANGUAGES, Config
from .pipeline import Pipeline

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"


def websocket_library() -> str | None:
    """Name of the WebSocket implementation uvicorn will use, if any.

    uvicorn serves WebSockets only when `websockets` or `wsproto` is
    installed. Without one it answers /ws with **404 and keeps serving the
    page**, so the UI loads perfectly and then never updates - a blank
    caption pane with no error anywhere. That is the worst possible failure
    for a live demo, and the test suite cannot catch it: Starlette's
    TestClient uses an in-process transport and never performs a real
    handshake, so every WebSocket test passes regardless.

    Hence an explicit check at startup, and a loud message.
    """
    for name in ("websockets", "wsproto"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return None


def create_app(cfg: Config):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    bus = EventBus()
    pipeline = Pipeline(cfg, bus)
    app = FastAPI(title="Sahaay", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    async def _startup() -> None:
        bus.bind_loop(asyncio.get_running_loop())
        # Model loading is seconds to tens of seconds and must not block the
        # event loop, or the page never renders and the user assumes a hang.
        asyncio.get_running_loop().run_in_executor(None, _safe_load)

    def _safe_load() -> None:
        try:
            pipeline.load_models()
        except Exception as exc:  # noqa: BLE001
            log.exception("model loading failed")
            bus.publish("error", message=f"Model loading failed: {exc}")

    # -- pages -------------------------------------------------------------

    @app.get("/")
    async def index():
        return FileResponse(UI_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    # -- control -----------------------------------------------------------

    @app.get("/api/status")
    async def status():
        return JSONResponse(
            {
                **pipeline.status(),
                "languages": [
                    {"code": k, "name": v["name"]} for k, v in SUPPORTED_LANGUAGES.items()
                ],
            }
        )

    @app.post("/api/start")
    async def start():
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, pipeline.start)
        except Exception as exc:  # noqa: BLE001
            log.exception("start failed")
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
        return JSONResponse({"ok": True, **pipeline.status()})

    @app.post("/api/stop")
    async def stop():
        loop = asyncio.get_running_loop()
        # stop() runs the summariser, which is an LLM call - definitely not
        # something to do on the event loop.
        notes = await loop.run_in_executor(None, pipeline.stop)
        return JSONResponse({"ok": True, "saved": notes is not None, **pipeline.status()})

    @app.post("/api/language/{code}")
    async def set_language(code: str):
        if code not in SUPPORTED_LANGUAGES:
            return JSONResponse({"ok": False, "error": "unknown language"}, status_code=400)
        pipeline.set_language(code)
        return JSONResponse({"ok": True, "target_language": code})

    @app.get("/api/sessions")
    async def sessions():
        d = cfg.sessions_dir
        if not d.exists():
            return JSONResponse({"sessions": []})
        items = [
            {"name": p.stem, "path": str(p), "modified": p.stat().st_mtime}
            for p in sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        ]
        return JSONResponse({"sessions": items[:50]})

    # -- live feed ---------------------------------------------------------

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        q = bus.subscribe()
        try:
            # Send current state immediately so a page that connects mid-
            # lecture renders the device badge without waiting for an event.
            await websocket.send_text(json.dumps({"kind": "status", **pipeline.status()}))
            while True:
                event = await q.get()
                await websocket.send_text(event.to_json())
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("websocket closed: %s", exc)
        finally:
            bus.unsubscribe(q)

    app.state.pipeline = pipeline
    app.state.bus = bus
    return app


def serve(cfg: Config) -> None:
    import uvicorn

    ws_library = websocket_library()
    if ws_library is None:
        # Refuse rather than serve a UI that can never update. A blank page
        # with no explanation costs far more than a failed start.
        raise SystemExit(
            "\n  No WebSocket library is installed, so live captions cannot be\n"
            "  delivered and the page would load but never update.\n\n"
            "    pip install websockets\n\n"
            "  (or reinstall dependencies: pip install -r requirements.txt)\n"
        )
    log.debug("websocket transport: %s", ws_library)

    app = create_app(cfg)
    url = f"http://{cfg.server.host}:{cfg.server.port}"
    log.info("Sahaay UI at %s", url)

    if cfg.server.open_browser:
        with contextlib.suppress(Exception):
            # Delay so the server is listening before the tab opens.
            import threading

            threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")
