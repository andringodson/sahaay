"""The orchestrator: audio in, captions out.

Three threads, deliberately:

* **capture** - reads the audio device and segments on pauses. Must never
  block, because blocking here means dropping real speech.
* **transcribe** - drains segments, runs Whisper, then translation, then
  publishes the caption.
* **glossary** - owned by GlossaryWorker, runs the LLM at low priority.

Splitting transcription from capture is what lets a slow segment (a long
sentence, a cold graph) absorb into the queue instead of gapping the audio.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import queue
import threading
import time

import numpy as np

from . import bus as ev
from .asr import create_asr
from .audio import create_source
from .bus import EventBus
from .config import Config
from .glossary import GlossaryWorker, GlossEntry
from .llm import create_llm
from .metrics import Metrics
from .notes import CaptionRecord, NotesWriter, SessionNotes
from .runtime import create_factory
from .translate import create_translator
from .vad import Segment, Segmenter, create_vad

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config, bus: EventBus | None = None):
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.metrics = Metrics()

        self.factory = create_factory(cfg.runtime, mock=cfg.mock)
        self.device = self.factory.report()
        log.info("device: %s", self.device.summary())

        self.captions: list[CaptionRecord] = []
        self.glossary_entries: list[GlossEntry] = []
        self.started_at: dt.datetime | None = None
        self.last_notes: SessionNotes | None = None

        self._segments: queue.Queue[Segment | None] = queue.Queue(maxsize=32)
        self._translations: queue.Queue[tuple[CaptionRecord, str] | None] = queue.Queue(
            maxsize=64
        )
        self._translate_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._running = False

        self._source = None
        self._asr = None
        self._translator = None
        self._segmenter: Segmenter | None = None
        self._glossary: GlossaryWorker | None = None
        self._notes_writer: NotesWriter | None = None

    # -- state -------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict:
        return {
            "running": self._running,
            "device": self.device.to_dict(),
            "captions": len(self.captions),
            "glossary": len(self.glossary_entries),
            "target_language": self.cfg.translate.target_language,
            "mock": self.cfg.mock,
            "metrics": self.metrics.snapshot(),
        }

    def _publish_status(self) -> None:
        self.bus.publish(ev.STATUS, **self.status())

    # -- lifecycle ---------------------------------------------------------

    def load_models(self) -> None:
        """Load every model up front.

        Lazy-loading the LLM on first use would stall the first glossary
        lookup by however long a 3B model takes to finalise its graph, which
        on the HTP is seconds. Better to pay it before the lecture starts.
        """
        models = self.cfg.models_dir

        self.bus.publish(ev.STATUS, stage="loading", detail="speech recognition", **self.status())
        self._asr = create_asr(models, self.factory, self.cfg.asr, mock=self.cfg.mock)

        self.bus.publish(ev.STATUS, stage="loading", detail="translation", **self.status())
        self._translator = create_translator(
            models, self.factory, self.cfg.translate, mock=self.cfg.mock
        )

        self.bus.publish(ev.STATUS, stage="loading", detail="language model", **self.status())
        llm = create_llm(
            models,
            self.cfg.glossary.model_id,
            mock=self.cfg.mock,
            candidates=self.cfg.glossary.candidates,
            npu_active=self.factory.npu_active,
        )

        self._glossary = GlossaryWorker(
            llm, self.cfg.glossary, self.cfg.translate.target_language, on_gloss=self._on_gloss
        )
        self._notes_writer = NotesWriter(llm, self.cfg.notes, self.cfg.sessions_dir)

        self.bus.publish(ev.STATUS, stage="warming", **self.status())
        self._warm_up()

        self.bus.publish(ev.STATUS, stage="ready", **self.status())

    def _warm_up(self) -> None:
        """Run one throwaway inference per stage before the lecture starts.

        Creating a session is not the same as executing one. The first run
        finalises the graph, and on the Hexagon NPU that costs seconds - the
        benchmark harness discards warm-up runs for exactly this reason. Left
        unpaid here, a student presses Start and the bill lands on their first
        sentence, which is the one moment the app cannot afford to be slow.

        Failures are ignored on purpose: a warm-up that cannot run is not a
        reason to refuse to start, and whatever broke will surface with a real
        error on the first real segment.
        """
        import numpy as np

        started = time.monotonic()
        quiet = np.zeros(int(self.cfg.audio.sample_rate * 1.0), dtype=np.float32)

        try:
            if self._asr is not None:
                self._asr.transcribe(quiet)
        except Exception as exc:  # noqa: BLE001
            log.debug("ASR warm-up skipped: %s", exc)

        try:
            if self._translator is not None:
                self._translator.translate("warm up", self.cfg.translate.target_language)
        except Exception as exc:  # noqa: BLE001
            log.debug("translator warm-up skipped: %s", exc)

        log.info("warm-up took %.1fs", time.monotonic() - started)

    def start(self) -> None:
        if self._running:
            return
        if self._asr is None:
            self.load_models()

        self.captions.clear()
        self.glossary_entries.clear()
        self.metrics.reset()
        self.bus.clear_history()
        self.started_at = dt.datetime.now()
        self._stop.clear()

        self._source = create_source(
            self.cfg.audio,
            mock=self.cfg.mock,
            audio_file=self.cfg.audio_file,
            rate=self.cfg.audio_rate,
        )
        self._source.start()

        vad = create_vad(self.cfg.models_dir, self.cfg.vad)
        self._segmenter = Segmenter(self.cfg.audio, self.cfg.vad, vad)

        assert self._glossary is not None
        self._glossary.start()

        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._transcribe_loop, name="transcribe", daemon=True),
        ]
        for t in self._threads:
            t.start()

        if self.cfg.translate.enabled and self.cfg.translate.concurrent:
            self._translations = queue.Queue(maxsize=64)
            self._translate_thread = threading.Thread(
                target=self._translate_loop, name="translate", daemon=True
            )
            self._translate_thread.start()

        self._running = True
        log.info("pipeline started on %s", self.device.provider_label)
        self._publish_status()

    def stop(self) -> SessionNotes | None:
        if not self._running:
            return self.last_notes
        self._running = False
        self._stop.set()

        if self._source:
            self._source.stop()

        # Flush any half-finished sentence before tearing down: pressing Stop
        # mid-sentence must not lose the sentence.
        tail = self._segmenter.finalize() if self._segmenter else None
        if tail is not None:
            with contextlib.suppress(queue.Full):
                self._segments.put_nowait(tail)

        with contextlib.suppress(queue.Full):
            self._segments.put_nowait(None)  # sentinel

        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()

        # Let queued translations finish before the notes are written, so
        # the session file carries them. Bounded: a long backlog on a slow
        # machine must not hold the Stop button for minutes.
        if self._translate_thread is not None:
            with contextlib.suppress(queue.Full):
                self._translations.put(None, timeout=1.0)
            self._translate_thread.join(timeout=20.0)
            self._translate_thread = None

        if self._glossary:
            self._glossary.stop()
            self.glossary_entries = list(self._glossary.entries)

        notes = self._finish_session()
        self._publish_status()
        return notes

    # -- threads -----------------------------------------------------------

    def _capture_loop(self) -> None:
        mock_tick = 0
        level_sent = 0.0
        while not self._stop.is_set():
            chunk = self._source.read(timeout=0.5) if self._source else None
            if chunk is None:
                continue

            self.metrics.add_audio(chunk.size / self.cfg.audio.sample_rate)

            # Input loudness, five times a second. The UI draws it as a meter
            # because every other signal the app gives is downstream of a
            # caption: with a muted output or the wrong loopback device the
            # screen stays empty and says nothing about why. A meter that
            # does not move is an answer.
            now = time.monotonic()
            if now - level_sent >= 0.2:
                level_sent = now
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                self.metrics.set_level(rms)
                self.bus.publish(ev.LEVEL, rms=round(rms, 5))

            if self.cfg.mock and self.cfg.audio_file is None:
                # Mock audio is silence, so the VAD would never fire. Emit a
                # synthetic segment on a timer instead so the full downstream
                # path - ASR, translation, glossary, UI - still exercises.
                mock_tick += 1
                if mock_tick % 90 == 0:  # ~2.9 s of 32 ms frames
                    t = mock_tick * 0.032
                    seg = Segment(
                        audio=np.zeros(int(self.cfg.audio.sample_rate * 2.5), dtype=np.float32),
                        start_s=t - 2.5,
                        end_s=t,
                        index=mock_tick // 90,
                    )
                    self._enqueue(seg)
                continue

            try:
                with self.metrics.timed("vad"):
                    segments = self._segmenter.push(chunk)
            except Exception as exc:  # noqa: BLE001
                log.warning("segmentation error: %s", exc)
                continue

            for seg in segments:
                self._enqueue(seg)

    def _enqueue(self, seg: Segment) -> None:
        try:
            self._segments.put_nowait(seg)
        except queue.Full:
            # The transcriber is behind. Dropping the *oldest* keeps captions
            # near the live edge, which is what a live captioner should do.
            try:
                self._segments.get_nowait()
                self._segments.put_nowait(seg)
                log.warning("transcriber behind; dropped an older segment")
            except queue.Empty:
                pass

    def _transcribe_loop(self) -> None:
        while True:
            try:
                seg = self._segments.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if seg is None:
                break
            seg, stop_after = self._merge_backlog(seg)
            try:
                self._handle_segment(seg)
            except Exception as exc:  # noqa: BLE001
                log.exception("segment failed")
                self.bus.publish(ev.ERROR, message=str(exc))
            if stop_after:
                break

    def _merge_backlog(self, seg: Segment) -> tuple[Segment, bool]:
        """Fold queued segments into this one when transcription is behind.

        Whisper's encoder runs a fixed 30 s window whatever it is given, so
        each separate call on a short sentence pays for 30 s of audio. When
        the queue backs up - which on CPU it does as soon as the glossary
        model is running, the measurement this project is built on - taking
        the backlog in one call is close to free throughput, where the old
        behaviour was to fall further behind until the queue overflowed and
        dropped sentences.

        Keeping up, the queue is empty and this returns ``seg`` untouched.
        Returns (segment, saw_sentinel).
        """
        limit = self.cfg.audio.merge_backlog_s
        if limit <= 0:
            return seg, False

        rate = self.cfg.audio.sample_rate
        gap = np.zeros(int(0.15 * rate), dtype=np.float32)
        parts, end_s, merged = [seg.audio], seg.end_s, 0
        length = seg.audio.size

        while True:
            try:
                nxt = self._segments.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                # Stop was pressed. Finish what we have, then honour it.
                if merged:
                    self.metrics.add_merges(merged)
                return self._joined(seg, parts, end_s), True
            if length + gap.size + nxt.audio.size > limit * rate:
                # Too long to fold in; it goes first next time round.
                self._requeue_front(nxt)
                break
            parts += [gap, nxt.audio]
            length += gap.size + nxt.audio.size
            end_s = nxt.end_s
            merged += 1

        if not merged:
            return seg, False
        self.metrics.add_merges(merged)
        return self._joined(seg, parts, end_s), False

    @staticmethod
    def _joined(seg: Segment, parts: list, end_s: float) -> Segment:
        if len(parts) == 1:
            return seg
        return Segment(
            audio=np.concatenate(parts), start_s=seg.start_s, end_s=end_s, index=seg.index
        )

    def _requeue_front(self, seg: Segment) -> None:
        """Put a segment back at the head of the queue, preserving order."""
        with self._segments.mutex:
            self._segments.queue.appendleft(seg)
            self._segments.not_empty.notify()

    def _handle_segment(self, seg: Segment) -> None:
        assert self._asr is not None and self._translator is not None

        self.bus.publish(ev.CAPTION_PARTIAL, text="...", index=seg.index)

        with self.metrics.timed("asr"):
            result = self._asr.transcribe(seg.audio)

        text = result.text.strip()
        if not text:
            return

        record = CaptionRecord(
            index=len(self.captions),
            text=text,
            language=result.language,
            start_s=seg.start_s,
            end_s=seg.end_s,
            asr_latency_ms=result.latency_ms,
            rtf=result.rtf,
        )

        self.bus.publish(
            ev.CAPTION,
            index=record.index,
            text=text,
            language=result.language,
            start_s=round(seg.start_s, 2),
            latency_ms=round(result.latency_ms, 1),
            rtf=round(result.rtf, 3),
            provider=result.provider,
        )

        self.captions.append(record)

        if self.cfg.translate.enabled:
            if self.cfg.translate.concurrent:
                # Hand off and move on. The next segment's English caption
                # no longer waits for this one's translation, which on CPU
                # took longer than the transcription did.
                try:
                    self._translations.put_nowait((record, text))
                except queue.Full:
                    self._translate(record, text)
            else:
                self._translate(record, text)

        if self._glossary:
            self._glossary.submit(text)

        self.bus.publish(ev.METRIC, **self.metrics.snapshot())

    def _translate(self, record: CaptionRecord, text: str) -> None:
        assert self._translator is not None
        with self.metrics.timed("translate"):
            tr = self._translator.translate(text, self.cfg.translate.target_language)
        record.translation = tr.text
        self.bus.publish(
            ev.TRANSLATION,
            index=record.index,
            text=tr.text,
            target_language=tr.target_language,
            latency_ms=round(tr.latency_ms, 1),
            passthrough=tr.passthrough,
            protected_terms=tr.protected_terms,
        )

    def _translate_loop(self) -> None:
        while True:
            try:
                item = self._translations.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._translate(*item)
            except Exception as exc:  # noqa: BLE001
                log.exception("translation failed")
                self.bus.publish(ev.ERROR, message=f"translation: {exc}")

    def _on_gloss(self, entry: GlossEntry) -> None:
        self.glossary_entries.append(entry)
        self.bus.publish(ev.GLOSS, **entry.to_dict())

    # -- session end -------------------------------------------------------

    def _finish_session(self) -> SessionNotes | None:
        if self._notes_writer is None or self.started_at is None:
            return None
        if not self.captions:
            log.info("no captions captured; nothing to save")
            return None

        self.bus.publish(ev.STATUS, stage="summarising", **self.status())
        notes = self._notes_writer.build(
            captions=self.captions,
            glossary=self.glossary_entries,
            target_language=self.cfg.translate.target_language,
            started_at=self.started_at,
            device=self.device.to_dict(),
            metrics=self.metrics.snapshot(),
        )
        path = self._notes_writer.save(notes)
        self.last_notes = notes

        self.bus.publish(
            ev.NOTES,
            title=notes.title,
            path=str(path),
            markdown=self._notes_writer.render_markdown(notes),
            summary=notes.summary_markdown,
            quiz=[{"q": q.question, "a": q.answer} for q in notes.quiz],
            glossary=[g.to_dict() for g in notes.glossary],
            duration_minutes=round(notes.duration_minutes, 1),
        )
        return notes

    def set_language(self, code: str) -> None:
        self.cfg.translate.target_language = code
        if self._glossary:
            self._glossary.target_language = code
        self._publish_status()


def build_pipeline(cfg: Config, bus: EventBus | None = None) -> Pipeline:
    return Pipeline(cfg, bus)
