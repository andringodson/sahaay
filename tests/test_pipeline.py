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
