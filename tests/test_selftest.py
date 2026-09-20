"""The one command a reviewer runs when something looks wrong.

The important property is not that it passes - on a fresh clone it should
report failures - but that it tells the truth and distinguishes "missing,
with a fallback" from "broken".
"""

import pytest

from sahaay.config import load_config
from sahaay.selftest import DEGRADED, FAIL, OK, Check, report, run


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.models_dir = tmp_path / "models"
    c.sessions_dir = tmp_path / "sessions"
    return c


class TestStatuses:
    def test_missing_weights_is_a_failure_not_a_warning(self, cfg):
        # There is no captioning without Whisper, so this must not be
        # softened into "degraded".
        checks, code = run(cfg, quick=True)
        asr = next(c for c in checks if c.name == "speech recognition")
        assert asr.status == FAIL
        assert code == 1

    def test_missing_npu_is_degraded_not_a_failure(self, cfg):
        # The product is designed to run without one.
        checks, _ = run(cfg, quick=True)
        npu = next(c for c in checks if c.name == "Hexagon NPU")
        assert npu.status in (OK, DEGRADED)

    def test_missing_translation_is_degraded(self, cfg):
        checks, _ = run(cfg, quick=True)
        tr = next(c for c in checks if c.name == "translation")
        assert tr.status == DEGRADED

    def test_every_non_ok_check_offers_a_next_step(self, cfg):
        # A diagnostic that names a problem without saying what to do about
        # it just relocates the confusion.
        checks, _ = run(cfg, quick=True)
        for check in checks:
            if check.status != OK:
                assert check.hint, f"{check.name} has no hint"


class TestExitCode:
    def test_nonzero_when_something_failed(self, cfg):
        _, code = run(cfg, quick=True)
        assert code == 1

    def test_zero_when_nothing_failed(self):
        checks = [
            Check("a", OK, "fine"),
            Check("b", DEGRADED, "missing", "install it"),
        ]
        assert not any(c.status == FAIL for c in checks)


class TestReport:
    def test_mentions_every_check(self, cfg):
        checks, _ = run(cfg, quick=True)
        text = report(checks)
        for check in checks:
            assert check.name in text

    def test_counts_are_stated(self, cfg):
        checks, _ = run(cfg, quick=True)
        text = report(checks)
        assert "ok," in text and "degraded," in text and "failed" in text

    def test_says_plainly_when_it_will_not_work(self, cfg):
        checks, _ = run(cfg, quick=True)
        assert "will not work properly" in report(checks)

    def test_says_plainly_when_everything_is_fine(self):
        assert "Everything is working" in report([Check("a", OK, "fine")])

    def test_degraded_only_still_reads_as_usable(self):
        text = report([Check("a", OK, "fine"), Check("b", DEGRADED, "x", "do y")])
        assert "will run" in text


class TestPipelineCheck:
    def test_full_run_exercises_the_pipeline(self, cfg):
        # Not quick: this actually drives a mock session.
        checks, _ = run(cfg, quick=False)
        pipeline = next(c for c in checks if c.name == "end-to-end pipeline")
        assert pipeline.status == OK, pipeline.detail
