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
    def test_the_device_report_never_says_qnn_is_missing_when_it_is_installed(self):
        """The failure this guards is a wrong instruction, not a wrong result.

        Falling back to CPU without a Hexagon NPU is correct. Saying "QNN not
        installed" while the package is installed is not: on a Snapdragon
        laptop that sends the user off to reinstall something they have.

        Asserting that QNN *must* register would be wrong here - the ARM64 CI
        runner is Microsoft Cobalt silicon with no Qualcomm NPU, so there is
        genuinely nothing for it to bind to. What has to hold on every machine
        is that the explanation is true.
        """
        from sahaay.runtime import SessionFactory

        pytest.importorskip("onnxruntime_qnn", reason="onnxruntime-qnn not installed")
        report = SessionFactory().report()

        if report.fallback_reason:
            assert "QNN not installed" not in report.fallback_reason, (
                f"onnxruntime-qnn is installed, but the report says: "
                f"{report.fallback_reason}"
            )

    def test_qnn_registration_is_visible_where_it_binds(self):
        """Where QNN does register, the app must see it.

        Asking get_available_providers() was wrong: that lists the providers
        compiled into the wheel, and QNN arrives at runtime as a plugin EP,
        which appears in get_ep_devices(). On x86 both lists contain it, so
        the wrong question went unnoticed for weeks.
        """
        from sahaay.runtime import SessionFactory, _qnn_registered

        SessionFactory()  # registration happens on import of the runtime
        pytest.importorskip("onnxruntime_qnn", reason="onnxruntime-qnn not installed")

        if "QNNExecutionProvider" in ort.get_available_providers():
            assert _qnn_registered(ort), (
                "QNN is in get_available_providers() but _qnn_registered() says no - "
                "the app would never activate the NPU path."
            )

    def test_reregistering_does_not_report_failure(self):
        """The second call must still hand back the HTP path.

        Re-registering raises "already registered". Treating that as an error
        made the runtime report the NPU unavailable on a machine where it was
        registered and working.
        """
        pytest.importorskip("onnxruntime_qnn", reason="onnxruntime-qnn not installed")
        from sahaay.runtime import _register_qnn_plugin

        first = _register_qnn_plugin(ort)
        second = _register_qnn_plugin(ort)
        assert first == second

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
