"""A tiny pub/sub bus connecting the pipeline threads to the UI.

The capture and inference stages run on worker threads; the web server runs
on an asyncio loop. Rather than sprinkling ``run_coroutine_threadsafe`` calls
through the pipeline, every stage just calls :meth:`EventBus.publish` and the
bus handles the hop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# Event kinds the UI understands. Keeping them as constants stops the
# JavaScript and the Python drifting apart.
CAPTION = "caption"           # a finalised caption line
CAPTION_PARTIAL = "partial"   # in-flight text, replaced by the next CAPTION
TRANSLATION = "translation"   # translation attached to an existing caption id
GLOSS = "gloss"               # a jargon term + explanation
STATUS = "status"             # pipeline/device state for the header
METRIC = "metric"             # per-stage latency sample
NOTES = "notes"               # end-of-session summary payload
ERROR = "error"


@dataclass
class Event:
    kind: str
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "ts": self.ts, **self.data}, ensure_ascii=False)


class EventBus:
    """Thread-safe fan-out to any number of asyncio subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Replayed to every new subscriber so a browser that connects late
        # still sees the transcript so far.
        self._history: list[Event] = []
        self._history_limit = 500

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
            backlog = list(self._history)
        for ev in backlog:
            with_suppress(q.put_nowait, ev)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, kind: str, **data: Any) -> Event:
        ev = Event(kind=kind, data=data)
        # Partial captions are transient by definition - replaying them to a
        # late subscriber would show text that has already been superseded.
        if kind not in {CAPTION_PARTIAL, METRIC}:
            with self._lock:
                self._history.append(ev)
                if len(self._history) > self._history_limit:
                    del self._history[: len(self._history) - self._history_limit]

        with self._lock:
            subscribers = list(self._subscribers)
        if not subscribers:
            return ev

        loop = self._loop
        if loop is None or not loop.is_running():
            # No UI attached yet (CLI mode); events still land in history.
            return ev

        for q in subscribers:
            loop.call_soon_threadsafe(with_suppress, q.put_nowait, ev)
        return ev

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


def with_suppress(fn, *args) -> None:
    """Drop events for a subscriber whose queue is full rather than blocking.

    A stalled browser tab must never be able to back-pressure the audio
    pipeline into dropping real speech.
    """
    try:
        fn(*args)
    except asyncio.QueueFull:
        log.debug("subscriber queue full; dropping event")
    except Exception as exc:  # noqa: BLE001
        log.debug("event delivery failed: %s", exc)
