"""Execution-provider selection and ONNX Runtime session construction.

This is the file that makes Sahaay a Snapdragon application rather than a
generic Python app. Every model in the pipeline is loaded through
:class:`SessionFactory`, so there is exactly one place that decides whether
work lands on the Hexagon NPU, the GPU, or the CPU - and exactly one place
that reports which one actually won.

Provider priority is QNN (Hexagon NPU) > DirectML (GPU) > CPU. The fallback
chain is deliberate: the app must stay usable on a reviewer's x86 laptop
while being fastest on the hardware it was built for.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RuntimeConfig

log = logging.getLogger(__name__)

# Human-readable names for the EP badge in the UI.
PROVIDER_LABELS = {
    "QNNExecutionProvider": "Hexagon NPU (QNN)",
    "DmlExecutionProvider": "GPU (DirectML)",
    "CPUExecutionProvider": "CPU",
}


def _short(provider: str) -> str:
    return provider.replace("ExecutionProvider", "")


def _import_ort():
    """Import onnxruntime and register Qualcomm's plugin EP.

    This registration step is not optional and is easy to miss. Since
    ONNX Runtime 1.23 the Qualcomm EP ships as a *plugin*: ``pip install
    onnxruntime-qnn`` lays down a separate ``onnxruntime_qnn`` package
    containing ``onnxruntime_providers_qnn.dll``, but onnxruntime does not
    load it on its own. Until ``register_execution_provider_library`` is
    called, ``get_available_providers()`` does not list QNN at all - so a
    provider-priority list that simply looks for "QNNExecutionProvider"
    silently falls through to CPU *on the Snapdragon device itself*, which
    is the exact failure this project could not afford.

    Verified locally: providers before registration were
    ``['AzureExecutionProvider', 'CPUExecutionProvider']``; after, QNN is
    present.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "onnxruntime is not installed. Run install.ps1, or:\n"
            "  pip install onnxruntime-qnn   (Windows, Python 3.11+)\n"
            "  pip install onnxruntime       (anything else)"
        ) from exc

    _register_qnn_plugin(ort)
    return ort


def _register_qnn_plugin(ort) -> str | None:  # noqa: ANN001
    """Register the QNN plugin EP. Returns the QnnHtp.dll path, if any.

    Safe to call repeatedly: re-registering the same name raises, and that
    is not an error worth propagating.
    """
    if "QNNExecutionProvider" in ort.get_available_providers():
        return _htp_path()

    try:
        import onnxruntime_qnn as oq  # type: ignore
    except ImportError:
        # Expected on non-Windows, or a plain onnxruntime install.
        log.debug("onnxruntime-qnn not installed; NPU path unavailable")
        return None

    if not hasattr(ort, "register_execution_provider_library"):
        log.warning(
            "onnxruntime %s predates plugin execution providers; "
            "upgrade to 1.23+ for Hexagon NPU support",
            getattr(ort, "__version__", "?"),
        )
        return None

    try:
        ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
        log.info("registered %s from %s", oq.get_ep_name(), oq.get_library_path())
    except Exception as exc:  # noqa: BLE001
        log.warning("could not register the QNN execution provider: %s", exc)
        return None
    return _htp_path()


def _htp_path() -> str | None:
    """Full path to QnnHtp.dll.

    Passing a bare "QnnHtp.dll" as backend_path relies on it being on PATH,
    which it is not when it lives inside a site-packages wheel. The package
    resolves the right amd64/arm64ec/arm64 subdirectory for us.
    """
    try:
        import onnxruntime_qnn as oq  # type: ignore

        path = oq.get_qnn_htp_path()
        return path if Path(path).exists() else None
    except Exception:  # noqa: BLE001
        return None


def _qnn_hardware_type(ort) -> str | None:  # noqa: ANN001
    """What hardware the registered QNN provider is actually bound to.

    Registering the plugin is not the same as having a Hexagon NPU. On an
    x86 box the QNN provider registers happily and then reports hardware
    type CPU. Reporting that as "NPU active" would be the kind of claim
    this project is specifically trying not to make.
    """
    if not hasattr(ort, "get_ep_devices"):
        return None
    try:
        for device in ort.get_ep_devices():
            if device.ep_name == "QNNExecutionProvider":
                return device.device.type.name  # 'NPU' | 'GPU' | 'CPU'
    except Exception:  # noqa: BLE001
        return None
    return None


@dataclass
class DeviceReport:
    """What the app is actually running on - shown in the UI and the README."""

    provider: str
    provider_label: str
    available_providers: list[str]
    ort_version: str
    machine: str
    processor: str
    is_arm64: bool
    npu_active: bool
    fallback_reason: str | None = None
    # What ORT says the QNN provider is bound to: 'NPU', 'CPU', 'GPU' or None.
    qnn_hardware: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_label": self.provider_label,
            "available_providers": self.available_providers,
            "ort_version": self.ort_version,
            "machine": self.machine,
            "processor": self.processor,
            "is_arm64": self.is_arm64,
            "npu_active": self.npu_active,
            "fallback_reason": self.fallback_reason,
            "qnn_hardware": self.qnn_hardware,
        }

    def summary(self) -> str:
        line = f"{self.provider_label}  |  onnxruntime {self.ort_version}  |  {self.machine}"
        if self.fallback_reason:
            line += f"\n  fallback: {self.fallback_reason}"
        return line


class SessionFactory:
    """Creates ORT InferenceSessions on the best available provider.

    One factory is shared by every stage of the pipeline so the ASR, the
    translator and the LLM all report against the same device.
    """

    def __init__(self, cfg: RuntimeConfig | None = None):
        self.cfg = cfg or RuntimeConfig()
        self._ort = _import_ort()
        self._available: list[str] = list(self._ort.get_available_providers())
        self._qnn_hardware = _qnn_hardware_type(self._ort)
        self._htp = _htp_path()
        self._chosen, self._reason = self._choose()

    # -- provider selection ------------------------------------------------

    def _choose(self) -> tuple[str, str | None]:
        """Walk the priority list, recording *why* each rejection happened.

        The reason string ends up in the UI and the benchmark header, so it
        has to distinguish "not installed" from "installed but there is no
        such hardware here" - they look identical from the provider list and
        mean completely different things.
        """
        skipped: list[str] = []

        for candidate in self.cfg.provider_priority:
            if candidate not in self._available:
                skipped.append(f"{_short(candidate)} not installed")
                continue

            # QNN registered but bound to CPU means no Hexagon on this box.
            # Using it anyway would be slower than the plain CPU provider and
            # would let the UI claim an NPU that is not there.
            if candidate == "QNNExecutionProvider" and self._qnn_hardware not in (None, "NPU"):
                log.info("QNN provider is bound to %s, not NPU; skipping it", self._qnn_hardware)
                skipped.append(
                    f"QNN present but bound to {self._qnn_hardware}, so there is no Hexagon NPU here"
                )
                continue

            if not skipped:
                return candidate, None
            return candidate, "; ".join(skipped)

        return "CPUExecutionProvider", "; ".join(skipped) or "no accelerated provider available"

    def _provider_options(self, provider: str) -> dict[str, Any]:
        """Per-provider tuning.

        The QNN options are the ones that matter for a lecture workload:
        short, repeated graph executions where we care about time-to-first
        token far more than throughput.
        """
        if provider == "QNNExecutionProvider":
            opts: dict[str, Any] = {
                # QnnHtp.dll targets the Hexagon Tensor Processor. Swapping
                # this for QnnCpu.dll is a useful A/B when debugging accuracy
                # differences between the NPU and reference execution.
                # Full path, not a bare filename: the DLL lives inside the
                # onnxruntime_qnn wheel and is not on PATH.
                "backend_path": self._htp or "QnnHtp.dll",
                "htp_performance_mode": self.cfg.htp_performance_mode,
                "htp_graph_finalization_optimization_mode": (
                    self.cfg.htp_graph_finalization_optimization_mode
                ),
                # Keeps the graph resident between calls. Without this, a
                # per-segment workload pays context-load cost every segment,
                # which dominates the measurement.
                "qnn_context_priority": "high",
            }
            if self.cfg.enable_htp_fp16_precision:
                opts["enable_htp_fp16_precision"] = "1"
            return opts
        if provider == "DmlExecutionProvider":
            return {}
        return {}

    # -- public API --------------------------------------------------------

    @property
    def provider(self) -> str:
        return self._chosen

    @property
    def npu_active(self) -> bool:
        """True only when work really lands on a Hexagon NPU.

        Deliberately stricter than "the QNN provider was selected": see
        :func:`_qnn_hardware_type`.
        """
        return self._chosen == "QNNExecutionProvider" and self._qnn_hardware in (None, "NPU")

    def report(self) -> DeviceReport:
        machine = platform.machine()
        return DeviceReport(
            provider=self._chosen,
            provider_label=PROVIDER_LABELS.get(self._chosen, self._chosen),
            available_providers=self._available,
            ort_version=getattr(self._ort, "__version__", "unknown"),
            machine=machine,
            processor=platform.processor() or "unknown",
            is_arm64=machine.lower() in {"arm64", "aarch64"},
            npu_active=self.npu_active,
            fallback_reason=self._reason,
            qnn_hardware=self._qnn_hardware,
        )

    def create(self, model_path: str | Path, *, provider: str | None = None):
        """Build an InferenceSession for ``model_path`` on the chosen provider.

        If the preferred provider rejects the model (a common, recoverable
        situation: a graph with ops the HTP backend does not implement), we
        retry on CPU rather than taking the whole app down. The stage that
        fell back is reported so the benchmark table stays honest.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run: python scripts/download_models.py --all"
            )

        provider = provider or self._chosen
        so = self._ort.SessionOptions()
        so.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.cfg.intra_op_threads:
            so.intra_op_num_threads = self.cfg.intra_op_threads

        providers = [provider]
        provider_options = [self._provider_options(provider)]
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        try:
            sess = self._ort.InferenceSession(
                str(model_path), sess_options=so, providers=providers, provider_options=provider_options
            )
        except Exception as exc:  # noqa: BLE001 - we want the fallback to be total
            log.warning("%s rejected %s (%s); retrying on CPU", provider, model_path.name, exc)
            sess = self._ort.InferenceSession(
                str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
            )

        actual = sess.get_providers()[0]
        if actual != provider:
            log.warning("%s fell back from %s to %s", model_path.name, provider, actual)
        log.info("loaded %s on %s", model_path.name, actual)
        return sess


class NullSessionFactory:
    """Stand-in used by ``--mock`` when onnxruntime is not installed at all.

    Mock mode exists so the UI, the event plumbing and the notes writer can
    be demoed on a machine with nothing downloaded. Requiring a 200 MB
    runtime just to see the interface would defeat that.
    """

    def __init__(self, cfg: RuntimeConfig | None = None):
        self.cfg = cfg or RuntimeConfig()

    @property
    def provider(self) -> str:
        return "MockProvider"

    @property
    def npu_active(self) -> bool:
        return False

    def report(self) -> DeviceReport:
        machine = platform.machine()
        return DeviceReport(
            provider="MockProvider",
            provider_label="Mock (no inference)",
            available_providers=[],
            ort_version="not installed",
            machine=machine,
            processor=platform.processor() or "unknown",
            is_arm64=machine.lower() in {"arm64", "aarch64"},
            npu_active=False,
            fallback_reason="running with --mock; no models are executed",
        )

    def create(self, model_path: str | Path, *, provider: str | None = None):
        raise RuntimeError("NullSessionFactory cannot create sessions (mock mode)")


def create_factory(cfg: RuntimeConfig | None = None, mock: bool = False):
    """Build the real factory, or a null one if mock mode has no runtime."""
    try:
        return SessionFactory(cfg)
    except RuntimeError:
        if mock:
            log.warning("onnxruntime not installed; mock mode will not run models")
            return NullSessionFactory(cfg)
        raise


def describe_device() -> DeviceReport:
    """Convenience for scripts and the ``--device`` CLI flag."""
    return SessionFactory().report()
