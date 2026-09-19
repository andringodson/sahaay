"""The jargon sidebar.

When the lecturer says "eigenvalue", a student who is following the lecture
in their second language needs one line telling them what it is - in their
own language, without leaving the lecture to search for it.

This is the stage that justifies the NPU. It runs a 3B model concurrently
with Whisper, continuously, for the length of a lecture. On CPU that either
starves the captions or flattens the battery; on the Hexagon NPU both models
stay resident and the glossary costs the user nothing they notice.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field

from .config import SUPPORTED_LANGUAGES, GlossaryConfig
from .llm import LlmBackend

log = logging.getLogger(__name__)


PROMPT = """You are helping a student follow a university lecture.

Below is a short excerpt of the lecture transcript. Find up to {n} technical \
terms a first-year student would not already know. Ignore ordinary words and \
ignore names of people.

For each term write ONE short line, in this exact format:
term :: explanation in {language}, max 18 words

If there are no technical terms, reply with the single word NONE.

<transcript>
{transcript}
</transcript>
"""


@dataclass
class GlossEntry:
    term: str
    explanation: str
    language: str
    source_line: str
    latency_ms: float = 0.0
    backend: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "explanation": self.explanation,
            "language": self.language,
            "source_line": self.source_line,
            "latency_ms": round(self.latency_ms, 1),
            "backend": self.backend,
        }


class GlossaryWorker:
    """Background consumer of caption lines that emits glossary entries.

    Runs on its own thread at low priority. The queue is deliberately small
    and lossy: if the LLM cannot keep up with the lecturer, the right
    behaviour is to skip old lines, not to build an ever-growing backlog and
    explain a term four minutes after it was spoken.
    """

    def __init__(self, llm: LlmBackend, cfg: GlossaryConfig, target_language: str, on_gloss=None):
        self.llm = llm
        self.cfg = cfg
        self.target_language = target_language
        self.on_gloss = on_gloss

        self._q: queue.Queue[str] = queue.Queue(maxsize=12)
        self._pending: list[str] = []
        self._seen: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.entries: list[GlossEntry] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="glossary", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def submit(self, line: str) -> None:
        if not self.cfg.enabled or not line.strip():
            return
        try:
            self._q.put_nowait(line)
        except queue.Full:
            # Drop the oldest so we stay near the live edge of the lecture.
            try:
                self._q.get_nowait()
                self._q.put_nowait(line)
            except queue.Empty:
                pass

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            self._pending.append(line)
            if len(self._pending) < self.cfg.batch_lines:
                continue
            self._process(self._pending)
            self._pending = []

        if self._pending:
            self._process(self._pending)

    def _process(self, lines: list[str]) -> None:
        if len(self._seen) >= self.cfg.max_terms_per_session:
            return

        language = SUPPORTED_LANGUAGES.get(self.target_language, {}).get("name", "English")
        prompt = PROMPT.format(
            n=3, language=language, transcript=" ".join(lines)
        )
        try:
            result = self.llm.generate(prompt, max_new_tokens=self.cfg.max_new_tokens)
        except Exception as exc:  # noqa: BLE001 - a glossary failure must never
            log.warning("glossary generation failed: %s", exc)  # stop the captions
            return

        for entry in self._parse(result.text, source_line=lines[-1]):
            key = entry.term.lower()
            if key in self._seen:
                continue
            self._seen.add(key)
            entry.latency_ms = result.latency_ms
            entry.backend = result.backend
            self.entries.append(entry)
            if self.on_gloss:
                self.on_gloss(entry)
            if len(self._seen) >= self.cfg.max_terms_per_session:
                break

    def _parse(self, text: str, source_line: str) -> list[GlossEntry]:
        out: list[GlossEntry] = []
        if not text or text.strip().upper().startswith("NONE"):
            return out

        for raw in text.splitlines():
            raw = raw.strip()
            if not raw or "::" not in raw:
                continue
            term, _, explanation = raw.partition("::")
            # Models like to prefix list items; strip the decoration.
            term = re.sub(r"^[\-\*\d\.\)\s]+", "", term).strip().strip('"*')
            explanation = explanation.strip().strip('"')
            if not term or not explanation or len(term) > 48:
                continue
            # Drop truncation artefacts. When generation hits the token
            # budget mid-word the last entry arrives as a fragment - observed
            # as "Characteristic Equation :: मैट्र" (5 characters). The
            # threshold stays low deliberately: "disorder" is a legitimate
            # eight-character gloss, and an earlier cut of 12 deleted it.
            if len(explanation) < 6:
                log.debug("dropping truncated gloss for %r: %r", term, explanation)
                continue
            out.append(
                GlossEntry(
                    term=term,
                    explanation=explanation,
                    language=self.target_language,
                    source_line=source_line,
                )
            )
        return out
