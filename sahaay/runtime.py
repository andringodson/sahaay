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

# Devices we have validated against, used only for reporting.
KNOWN_SNAPDRAGON_MARKERS = ("Snapdragon", "X Elite", "X Plus", "X2", "Oryon", "ARM64")


def _import_ort():
    """Import onnxruntime, preferring the QNN build.

    ``onnxruntime-qnn`` installs under the same ``onnxruntime`` module name,
    so a plain import picks up whichever wheel is installed. We only need to
    surface a clear error when neither is present.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "onnxruntime is not installed. Run install.ps1, or:\n"
            "  pip install onnxruntime-qnn   (Windows, Python 3.11+)\n"
            "  pip install onnxruntime       (anything else)"
        ) from exc
    return ort


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
        self._chosen, self._reason = self._choose()

    # -- provider selection ------------------------------------------------

    def _choose(self) -> tuple[str, str | None]:
        for candidate in self.cfg.provider_priority:
            if candidate in self._available:
                if candidate == self.cfg.provider_priority[0]:
                    return candidate, None
                missing = self.cfg.provider_priority[: self.cfg.provider_priority.index(candidate)]
                return candidate, f"{', '.join(missing)} not available in this onnxruntime build"
        return (
            "CPUExecutionProvider",
            f"none of {self.cfg.provider_priority} available; available: {self._available}",
        )

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
                "backend_path": "QnnHtp.dll",
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
        return self._chosen == "QNNExecutionProvider"

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
