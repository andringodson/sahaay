"""Central configuration.

Values resolve in this order: explicit argument > environment variable >
``sahaay.json`` next to the repo root > the defaults below. Keeping this in
one place means the benchmark harness and the app can never drift apart on
things like sample rate or chunk length.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("SAHAAY_MODELS_DIR", REPO_ROOT / "models"))
SESSIONS_DIR = Path(os.environ.get("SAHAAY_SESSIONS_DIR", REPO_ROOT / "sessions"))

# Whisper is trained on 16 kHz mono; everything upstream matches it so we
# never resample twice.
SAMPLE_RATE = 16_000
FRAME_MS = 32  # Silero VAD consumes 512-sample frames at 16 kHz.
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


# Languages we can translate captions into. Codes are (display, NLLB code,
# Whisper hint). Whisper's own language id is used for the *source* side.
SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "hi": {"name": "हिन्दी (Hindi)", "nllb": "hin_Deva", "whisper": "hi"},
    "ta": {"name": "தமிழ் (Tamil)", "nllb": "tam_Taml", "whisper": "ta"},
    "te": {"name": "తెలుగు (Telugu)", "nllb": "tel_Telu", "whisper": "te"},
    "kn": {"name": "ಕನ್ನಡ (Kannada)", "nllb": "kan_Knda", "whisper": "kn"},
    "ml": {"name": "മലയാളം (Malayalam)", "nllb": "mal_Mlym", "whisper": "ml"},
    "bn": {"name": "বাংলা (Bengali)", "nllb": "ben_Beng", "whisper": "bn"},
    "mr": {"name": "मराठी (Marathi)", "nllb": "mar_Deva", "whisper": "mr"},
    "gu": {"name": "ગુજરાતી (Gujarati)", "nllb": "guj_Gujr", "whisper": "gu"},
    "en": {"name": "English", "nllb": "eng_Latn", "whisper": "en"},
}


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    # Capture the speakers (lecture playing over Zoom/YouTube) as well as the
    # mic (a professor in the room). Either can be disabled from the UI.
    capture_loopback: bool = True
    capture_microphone: bool = True
    # A segment is flushed to the ASR when the speaker pauses this long, or
    # when it reaches max_segment_s, whichever comes first. The cap matters:
    # Whisper's receptive field is 30 s and quality falls off a cliff past it.
    silence_flush_ms: int = 700
    min_segment_s: float = 1.0
    max_segment_s: float = 12.0


@dataclass
class VadConfig:
    enabled: bool = True
    threshold: float = 0.5
    # Keep a little audio from before the trigger so we don't clip the first
    # consonant of a word.
    speech_pad_ms: int = 200


@dataclass
class AsrConfig:
    # Directory under models/ holding the Whisper encoder+decoder.
    model_id: str = "whisper_small_quantized"
    # None = let Whisper detect. Indian lecture audio is code-mixed, so
    # forcing a language usually hurts; detection per segment is better.
    language: str | None = None
    beam_size: int = 1  # greedy - we are latency bound, not quality bound
    max_decode_tokens: int = 220


@dataclass
class TranslateConfig:
    enabled: bool = True
    model_id: str = "nllb_200_distilled_600m_int8"
    target_language: str = "hi"
    max_tokens: int = 256


@dataclass
class GlossaryConfig:
    """The jargon sidebar.

    This runs the LLM at *low* priority alongside ASR. On the NPU both models
    stay resident; on CPU fallback we throttle hard so captions never stall.
    """

    enabled: bool = True
    model_id: str = "llama_3_2_3b_instruct_hexagon"
    # Don't ask the LLM about every caption line - batch a few together.
    batch_lines: int = 3
    # Never re-explain a term inside one session.
    max_terms_per_session: int = 60
    # Indic scripts cost far more tokens per character than English, so a
    # budget tuned on English output truncates the last entry of every batch.
    # Measured: three Hindi glosses need ~200 tokens, not 160.
    max_new_tokens: int = 256


@dataclass
class NotesConfig:
    enabled: bool = True
    quiz_questions: int = 5


@dataclass
class RuntimeConfig:
    """Execution-provider preference.

    ``provider_priority`` is walked in order and the first available one
    wins. Force a specific provider with SAHAAY_PROVIDER=CPUExecutionProvider
    to produce the CPU column of the benchmark table.
    """

    provider_priority: list[str] = field(
        default_factory=lambda: [
            "QNNExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    # Sustained transcription wants "burst" for short bursty graphs; the
    # Qualcomm docs recommend it for latency-sensitive workloads.
    htp_performance_mode: str = "burst"
    htp_graph_finalization_optimization_mode: str = "3"
    enable_htp_fp16_precision: bool = True
    intra_op_threads: int = 0  # 0 = let ORT decide


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8756
    open_browser: bool = True


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    glossary: GlossaryConfig = field(default_factory=GlossaryConfig)
    notes: NotesConfig = field(default_factory=NotesConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    models_dir: Path = field(default_factory=lambda: MODELS_DIR)
    sessions_dir: Path = field(default_factory=lambda: SESSIONS_DIR)

    # When no model weights are present the pipeline still runs end to end
    # using a scripted transcript. This keeps the UI and the event plumbing
    # developable (and demoable) on a machine with no models downloaded.
    mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["models_dir"] = str(self.models_dir)
        d["sessions_dir"] = str(self.sessions_dir)
        return d


def _apply_env(cfg: Config) -> Config:
    if v := os.environ.get("SAHAAY_PROVIDER"):
        # Forcing a provider means *only* that provider plus CPU as a floor,
        # otherwise ORT would silently promote us back to the NPU.
        cfg.runtime.provider_priority = [v] if v == "CPUExecutionProvider" else [v, "CPUExecutionProvider"]
    if v := os.environ.get("SAHAAY_TARGET_LANG"):
        cfg.translate.target_language = v
    if os.environ.get("SAHAAY_MOCK") in {"1", "true", "True"}:
        cfg.mock = True
    if v := os.environ.get("SAHAAY_PORT"):
        cfg.server.port = int(v)
    return cfg


def load_config(path: Path | None = None) -> Config:
    """Build a Config from defaults, then sahaay.json, then the environment."""
    cfg = Config()
    path = path or REPO_ROOT / "sahaay.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for section, values in raw.items():
            target = getattr(cfg, section, None)
            if target is None:
                continue
            if isinstance(values, dict):
                for k, v in values.items():
                    if hasattr(target, k):
                        setattr(target, k, v)
            else:
                setattr(cfg, section, values)
    return _apply_env(cfg)
