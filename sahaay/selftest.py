"""``sahaay --selftest`` - one command that says what works and what does not.

Written for someone who has just cloned the repo and wants to know, in under
a minute, whether it is working and what it will do on their machine. A
reviewer should not have to read the source to find out that translation is
unavailable because they skipped a download.

Every check reports one of three things, and the distinction is the point:

* **ok**      - working
* **degraded** - missing, and the app has a documented fallback
* **fail**    - broken, and the app will not work properly

A missing NPU is *degraded*, not *fail*: the product is designed to run
without one. Missing Whisper weights are *fail*, because there is no
captioning without them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Config, resolve_model_id

OK, DEGRADED, FAIL = "ok", "degraded", "fail"

SYMBOL = {OK: "+", DEGRADED: "~", FAIL: "x"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


def _runtime(cfg: Config) -> list[Check]:
    out: list[Check] = []
    try:
        from .runtime import SessionFactory

        factory = SessionFactory(cfg.runtime)
        report = factory.report()
    except RuntimeError as exc:
        return [
            Check("onnxruntime", FAIL, str(exc).splitlines()[0],
                  "pip install onnxruntime-qnn  (Windows, Python 3.11+)")
        ]

    out.append(Check("onnxruntime", OK, f"version {report.ort_version}"))

    if report.npu_active:
        out.append(Check("Hexagon NPU", OK, "active - models run on the NPU"))
    else:
        detail = report.fallback_reason or "not available"
        out.append(Check(
            "Hexagon NPU", DEGRADED, f"{report.provider_label} ({detail})",
            "Expected on any non-Snapdragon machine; see docs/HARDWARE.md",
        ))
    return out


def _models(cfg: Config) -> list[Check]:
    out: list[Check] = []

    def present(model_id: str) -> bool:
        path = cfg.models_dir / model_id
        return path.exists() and any(path.rglob("*.onnx"))

    asr_id = resolve_model_id(cfg.models_dir, cfg.asr.model_id, cfg.asr.candidates)
    if present(asr_id):
        out.append(Check("speech recognition", OK, asr_id))
    else:
        out.append(Check(
            "speech recognition", FAIL, "no Whisper weights found",
            "python scripts/download_models.py --asr",
        ))

    from .vad import find_silero

    if find_silero(cfg.models_dir):
        out.append(Check("voice activity detection", OK, "Silero"))
    else:
        out.append(Check(
            "voice activity detection", DEGRADED, "falling back to energy gating",
            "python scripts/download_models.py --vad",
        ))

    tr_id = resolve_model_id(cfg.models_dir, cfg.translate.model_id, cfg.translate.candidates)
    if present(tr_id):
        out.append(Check("translation", OK, tr_id))
    else:
        out.append(Check(
            "translation", DEGRADED, "captions will not be translated",
            "python scripts/download_models.py --translate",
        ))

    from .llm import find_genai_model, order_candidates

    npu = any(c.name == "Hexagon NPU" and c.status == OK for c in _RUNTIME_CACHE)
    llm_id = resolve_model_id(
        cfg.models_dir, cfg.glossary.model_id,
        order_candidates(cfg.glossary.candidates, npu),
    )
    if find_genai_model(cfg.models_dir / llm_id):
        out.append(Check("glossary language model", OK, llm_id))
    else:
        out.append(Check(
            "glossary language model", DEGRADED, "using the seeded heuristic glossary",
            "python scripts/download_models.py --llm",
        ))
    return out


def _transport() -> list[Check]:
    from .server import websocket_library

    library = websocket_library()
    if library:
        return [Check("live caption transport", OK, library)]
    return [Check(
        "live caption transport", FAIL,
        "uvicorn has no WebSocket library; the page would load but never update",
        "pip install websockets",
    )]


def _audio(cfg: Config) -> list[Check]:
    try:
        from .audio import WasapiSource

        devices = WasapiSource.list_devices()
    except Exception as exc:  # noqa: BLE001
        return [Check("audio capture", DEGRADED, f"unavailable ({exc})",
                      "pip install pyaudiowpatch  (Windows only)")]

    if not devices:
        return [Check("audio capture", DEGRADED, "no input devices found",
                      "pip install pyaudiowpatch, or use --mock")]

    loopback = sum(1 for d in devices if d.is_loopback)
    return [Check(
        "audio capture", OK,
        f"{len(devices)} input devices, {loopback} loopback",
    )]


def _pipeline(cfg: Config) -> list[Check]:
    """Run a short mock session end to end."""
    from .pipeline import Pipeline

    mock_cfg = Config(**{**cfg.__dict__})
    mock_cfg.mock = True
    try:
        pipeline = Pipeline(mock_cfg)
        pipeline.load_models()
        pipeline.start()
        deadline = time.time() + 25
        while time.time() < deadline and len(pipeline.captions) < 2:
            time.sleep(0.25)
        captions = len(pipeline.captions)
        pipeline.stop()
    except Exception as exc:  # noqa: BLE001
        return [Check("end-to-end pipeline", FAIL, f"{type(exc).__name__}: {exc}")]

    if captions >= 2:
        return [Check("end-to-end pipeline", OK, f"{captions} captions in mock mode")]
    return [Check("end-to-end pipeline", FAIL, f"only {captions} captions produced")]


_RUNTIME_CACHE: list[Check] = []


def run(cfg: Config, quick: bool = False) -> tuple[list[Check], int]:
    """Run every check. Returns the checks and a process exit code."""
    global _RUNTIME_CACHE

    # The report is the output. Interleaving it with pipeline INFO lines
    # makes the one thing this command exists to produce hard to read.
    import logging

    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.ERROR)

    try:
        checks: list[Check] = []
        _RUNTIME_CACHE = _runtime(cfg)
        checks += _RUNTIME_CACHE
        checks += _transport()
        checks += _models(cfg)
        checks += _audio(cfg)
        if not quick:
            checks += _pipeline(cfg)
    finally:
        root.setLevel(previous)

    failed = sum(1 for c in checks if c.status == FAIL)
    return checks, (1 if failed else 0)


def report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = ["", "  Sahaay self-test", "  " + "-" * (width + 40)]
    for c in checks:
        lines.append(f"  {SYMBOL[c.status]} {c.name.ljust(width)}  {c.detail}")
        if c.hint and c.status != OK:
            lines.append(f"    {' ' * width}  -> {c.hint}")

    ok = sum(1 for c in checks if c.status == OK)
    degraded = sum(1 for c in checks if c.status == DEGRADED)
    failed = sum(1 for c in checks if c.status == FAIL)

    lines += ["", f"  {ok} ok, {degraded} degraded, {failed} failed"]
    if failed:
        lines.append("  Sahaay will not work properly until the failures above are fixed.")
    elif degraded:
        lines.append("  Sahaay will run. Degraded items have documented fallbacks.")
    else:
        lines.append("  Everything is working.")
    lines.append("")
    return "\n".join(lines)
