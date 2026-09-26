"""End-to-end pipeline, metrics and notes.

Runs in mock mode so CI needs no model weights and no audio device.
"""

import datetime as dt
import time

import pytest

from sahaay import bus as ev
from sahaay.bus import EventBus
from sahaay.config import NotesConfig, load_config
from sahaay.glossary import GlossEntry
from sahaay.llm import HeuristicLlm
from sahaay.metrics import Metrics
from sahaay.notes import CaptionRecord, NotesWriter
from sahaay.pipeline import Pipeline


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.mock = True
    c.sessions_dir = tmp_path / "sessions"
    c.models_dir = tmp_path / "models"
    return c


class TestEventBus:
    def test_history_replays_to_a_late_subscriber(self):
        bus = EventBus()
        bus.publish(ev.CAPTION, text="first", index=0)
        # A browser opening mid-lecture should still see the transcript.
        assert len(bus._history) == 1

    def test_partials_are_not_replayed(self):
        # They have already been superseded by the time anyone reconnects.
        bus = EventBus()
        bus.publish(ev.CAPTION_PARTIAL, text="...", index=0)
        assert bus._history == []

    def test_history_is_bounded(self):
        bus = EventBus()
        bus._history_limit = 10
        for i in range(50):
            bus.publish(ev.CAPTION, text=str(i), index=i)
        assert len(bus._history) == 10

    def test_publish_without_subscribers_does_not_raise(self):
        EventBus().publish(ev.CAPTION, text="x", index=0)


class TestMetrics:
    def test_percentiles(self):
        m = Metrics()
        for v in range(1, 101):
            m.record("asr", float(v))
        s = m.stats("asr")
        assert s.count == 100
        assert s.min_ms == 1.0
        assert 90 <= s.p95_ms <= 100

    def test_realtime_factor(self):
        m = Metrics()
        m.add_audio(10.0)
        m.record("asr", 2000.0)  # 2 s of compute for 10 s of audio
        assert m.realtime_factor == pytest.approx(0.2)

    def test_unknown_stage_is_none(self):
        assert Metrics().stats("nope") is None

    def test_window_is_bounded(self):
        m = Metrics(window=20)
        for i in range(200):
            m.record("asr", float(i))
        # Count keeps rising; the sample buffer must not.
        assert m.stats("asr").count == 200
        assert len(m._samples["asr"]) == 20


class TestNotes:
    def writer(self, tmp_path):
        return NotesWriter(HeuristicLlm(), NotesConfig(), tmp_path / "sessions")

    def captions(self):
        return [
            CaptionRecord(index=0, text="Today we cover eigenvalue and entropy.",
                          translation="aaj hum padhenge", start_s=0.0, end_s=3.0),
            CaptionRecord(index=1, text="The determinant must be zero.",
                          translation="determinant zero hona chahiye", start_s=3.0, end_s=6.0),
        ]

    def test_builds_and_saves(self, tmp_path):
        w = self.writer(tmp_path)
        notes = w.build(self.captions(), [], "hi", dt.datetime.now())
        path = w.save(notes)
        assert path.exists()
        assert path.with_suffix(".json").exists()

    def test_markdown_contains_the_transcript(self, tmp_path):
        w = self.writer(tmp_path)
        md = w.render_markdown(w.build(self.captions(), [], "hi", dt.datetime.now()))
        assert "eigenvalue" in md
        assert "## Transcript" in md

    def test_glossary_is_rendered(self, tmp_path):
        w = self.writer(tmp_path)
        gloss = [GlossEntry(term="entropy", explanation="disorder", language="hi", source_line="x")]
        md = w.render_markdown(w.build(self.captions(), gloss, "hi", dt.datetime.now()))
        assert "## Glossary" in md and "entropy" in md

    def test_empty_session_does_not_crash(self, tmp_path):
        w = self.writer(tmp_path)
        notes = w.build([], [], "hi", dt.datetime.now())
        assert "No speech" in notes.summary_markdown

    def test_quiz_parsing(self, tmp_path):
        w = self.writer(tmp_path)
        items = w._parse_quiz("Q :: What is entropy? :: A measure of disorder.\ngarbage line")
        assert len(items) == 1
        assert items[0].answer.startswith("A measure")

    def test_long_transcript_is_excerpted_head_and_tail(self, tmp_path):
        w = self.writer(tmp_path)
        text = "A" * 4000 + "MIDDLE" + "B" * 4000
        out = w._excerpt(text, budget_chars=1000)
        assert "omitted" in out
        assert len(out) < len(text)


class TestPipelineMock:
    def test_runs_end_to_end(self, cfg):
        p = Pipeline(cfg)
        p.load_models()
        p.start()
        assert p.running

        deadline = time.time() + 20
        while time.time() < deadline and len(p.captions) < 3:
            time.sleep(0.25)

        notes = p.stop()
        assert not p.running
        assert len(p.captions) >= 3, "mock pipeline produced no captions"
        assert notes is not None
        assert notes.title

    def test_status_is_json_safe(self, cfg):
        import json

        p = Pipeline(cfg)
        json.dumps(p.status())  # the UI sends this over the wire

    def test_stop_without_start_is_safe(self, cfg):
        assert Pipeline(cfg).stop() is None

    def test_language_switch(self, cfg):
        p = Pipeline(cfg)
        p.load_models()
        p.set_language("ta")
        assert p.cfg.translate.target_language == "ta"
        assert p._glossary.target_language == "ta"


def _seg(seconds: float, index: int, start: float, rate: int = 16_000):
    import numpy as np

    from sahaay.vad import Segment

    return Segment(
        audio=np.full(int(seconds * rate), 0.1, dtype=np.float32),
        start_s=start, end_s=start + seconds, index=index,
    )


class TestBacklogMerge:
    """Behind? Take the queue in one Whisper call rather than one each.

    The encoder runs a fixed 30 s window whatever it is given, so separate
    calls on short queued sentences pay for 30 s apiece. On CPU the queue
    backs up as soon as the glossary model runs - the measurement this
    project is built on - and the old behaviour was to fall further behind
    until the queue overflowed and dropped sentences.
    """

    def test_a_queued_backlog_becomes_one_segment(self, cfg):
        p = Pipeline(cfg)
        first = _seg(4.0, 0, 0.0)
        p._segments.put(_seg(3.0, 1, 4.5))
        p._segments.put(_seg(5.0, 2, 8.0))

        merged, stop = p._merge_backlog(first)

        assert not stop
        assert merged.index == 0
        assert merged.start_s == 0.0
        assert merged.end_s == 13.0, "the merged segment must end where the last one did"
        # every sample of all three, plus two short breaths between them
        assert merged.audio.size == int((4 + 3 + 5 + 2 * 0.15) * 16_000)
        assert p._segments.empty()
        assert p.metrics.snapshot()["merged_segments"] == 2

    def test_keeping_up_changes_nothing(self, cfg):
        p = Pipeline(cfg)
        seg = _seg(4.0, 0, 0.0)
        merged, stop = p._merge_backlog(seg)
        assert merged is seg and not stop

    def test_it_never_exceeds_whispers_window(self, cfg):
        p = Pipeline(cfg)
        p._segments.put(_seg(10.0, 1, 12.0))
        p._segments.put(_seg(10.0, 2, 22.0))
        p._segments.put(_seg(10.0, 3, 32.0))

        merged, _ = p._merge_backlog(_seg(10.0, 0, 0.0))
        assert merged.audio.size / 16_000 <= cfg.audio.merge_backlog_s
        # what did not fit waits at the front, in order
        leftover = p._segments.get_nowait()
        assert leftover.index == 2, "order was lost putting a segment back"

    def test_stop_is_honoured_mid_merge(self, cfg):
        p = Pipeline(cfg)
        p._segments.put(_seg(3.0, 1, 4.0))
        p._segments.put(None)
        merged, stop = p._merge_backlog(_seg(3.0, 0, 0.0))
        assert stop, "the Stop sentinel was swallowed"
        assert merged.end_s == 7.0, "the segment before Stop was not kept"

    def test_it_can_be_switched_off(self, cfg):
        cfg.audio.merge_backlog_s = 0
        p = Pipeline(cfg)
        p._segments.put(_seg(3.0, 1, 4.0))
        seg = _seg(3.0, 0, 0.0)
        merged, _ = p._merge_backlog(seg)
        assert merged is seg and p._segments.qsize() == 1


class TestConcurrentTranslation:
    """Translation runs beside transcription, not after it.

    On CPU a translation took longer than the transcription it followed, and
    the next sentence's English caption waited for it.
    """

    def _run(self, cfg, concurrent: bool):
        cfg.translate.concurrent = concurrent
        p = Pipeline(cfg)
        seen = []
        original = p.bus.publish

        def tap(kind, **data):
            seen.append(kind)
            return original(kind, **data)

        p.bus.publish = tap
        p.load_models()
        p.start()
        deadline = time.time() + 20
        while time.time() < deadline and len(p.captions) < 3:
            time.sleep(0.25)
        notes = p.stop()
        return p, seen, notes

    def test_every_caption_still_gets_its_translation(self, cfg):
        p, seen, _ = self._run(cfg, concurrent=True)
        assert len(p.captions) >= 3
        assert seen.count(ev.TRANSLATION) == seen.count(ev.CAPTION), (
            "a translation was lost when it moved to its own thread"
        )

    def test_translations_land_before_the_notes_are_written(self, cfg):
        p, _, notes = self._run(cfg, concurrent=True)
        assert notes is not None
        assert all(r.translation is not None for r in p.captions), (
            "Stop wrote the notes before the translation thread drained"
        )

    def test_the_old_serial_path_still_works(self, cfg):
        p, seen, _ = self._run(cfg, concurrent=False)
        assert seen.count(ev.TRANSLATION) == seen.count(ev.CAPTION)
