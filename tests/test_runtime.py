"""Execution-provider selection.

These tests encode the lesson that cost the most time in this project: the
QNN plugin EP must be explicitly registered, and being registered is not the
same as having a Hexagon NPU. Both mistakes fail silently and produce a
product that claims NPU acceleration while running on CPU.
"""

import pytest

from sahaay.config import RuntimeConfig
from sahaay.runtime import PROVIDER_LABELS, NullSessionFactory, create_factory

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


class TestRegistration:
    def test_qnn_is_registered_when_the_package_is_present(self):
        from sahaay.runtime import SessionFactory

        SessionFactory()  # registration happens on import of the runtime
        oq = pytest.importorskip("onnxruntime_qnn", reason="onnxruntime-qnn not installed")
        assert oq.get_ep_name() in ort.get_available_providers(), (
            "QNN did not register. Without this the NPU path can never activate, "
            "even on a Snapdragon device."
        )

    def test_registration_is_idempotent(self):
        from sahaay.runtime import SessionFactory

        # The app builds factories freely; a second one must not explode on
        # re-registering the same library.
        SessionFactory()
        SessionFactory()


class TestSelection:
    def test_forced_cpu_is_honoured(self):
        from sahaay.runtime import SessionFactory

        f = SessionFactory(RuntimeConfig(provider_priority=["CPUExecutionProvider"]))
        assert f.provider == "CPUExecutionProvider"
        assert f.npu_active is False

    def test_never_claims_npu_without_npu_hardware(self):
        from sahaay.runtime import SessionFactory

        f = SessionFactory()
        report = f.report()
        if report.qnn_hardware not in (None, "NPU"):
            # QNN registered but bound to CPU/GPU: must not be selected and
            # must not be reported as an NPU.
            assert report.provider != "QNNExecutionProvider"
            assert report.npu_active is False

    def test_npu_active_implies_qnn_selected(self):
        from sahaay.runtime import SessionFactory

        report = SessionFactory().report()
        if report.npu_active:
            assert report.provider == "QNNExecutionProvider"

    def test_falls_back_rather_than_raising(self):
        from sahaay.runtime import SessionFactory

        f = SessionFactory(RuntimeConfig(provider_priority=["NoSuchExecutionProvider"]))
        assert f.provider == "CPUExecutionProvider"
        assert f.report().fallback_reason


class TestReport:
    def test_is_json_safe(self):
        import json

        from sahaay.runtime import SessionFactory

        json.dumps(SessionFactory().report().to_dict())

    def test_has_a_human_label(self):
        from sahaay.runtime import SessionFactory

        report = SessionFactory().report()
        assert report.provider_label
        if report.provider in PROVIDER_LABELS:
            assert report.provider_label == PROVIDER_LABELS[report.provider]

    def test_summary_mentions_the_fallback(self):
        from sahaay.runtime import SessionFactory

        f = SessionFactory(RuntimeConfig(provider_priority=["NoSuchExecutionProvider"]))
        assert "fallback" in f.report().summary()

    def test_missing_model_gives_an_actionable_error(self, tmp_path):
        from sahaay.runtime import SessionFactory

        with pytest.raises(FileNotFoundError, match="download_models"):
            SessionFactory().create(tmp_path / "nope.onnx")


class TestNullFactory:
    def test_reports_mock_without_claiming_npu(self):
        f = NullSessionFactory()
        assert f.npu_active is False
        assert f.report().provider == "MockProvider"

    def test_create_factory_falls_back_in_mock_mode(self):
        assert create_factory(mock=True) is not None
