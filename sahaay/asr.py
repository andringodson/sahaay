"""Whisper speech recognition on the Hexagon NPU.

Two Whisper packagings exist in the wild and we support both, because the
one you get depends on how you obtained the model:

* **Qualcomm AI Hub** ships ``WhisperEncoder`` / ``WhisperDecoder`` as
  separate graphs, pre-quantised to w8a16 and compiled to a QNN context
  binary for Snapdragon X Elite. This is the fast path and the one the
  benchmark numbers come from.
* **Hugging Face Optimum** exports ``encoder_model.onnx`` plus
  ``decoder_model_merged.onnx``. Slower, but it runs anywhere, which is what
  lets a reviewer on an x86 laptop reproduce the pipeline.

Rather than branching on a config flag, we introspect the graph's input
names and adapt. Model packaging changes between AI Hub releases and a
hard-coded assumption is the single most likely thing to rot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import AsrConfig
from .features import log_mel_spectrogram
from .runtime import SessionFactory

log = logging.getLogger(__name__)

# Whisper special tokens (multilingual vocabulary).
SOT = "<|startoftranscript|>"
EOT = "<|endoftext|>"
NO_TIMESTAMPS = "<|notimestamps|>"
TRANSCRIBE = "<|transcribe|>"


@dataclass
class AsrResult:
    text: str
    language: str | None
    duration_s: float
    latency_ms: float
    # Real-time factor: <1.0 means we transcribe faster than audio arrives,
    # which is the bar a live captioner has to clear.
    rtf: float = 0.0
    provider: str = ""
    tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class WhisperTokenizer:
    """Wraps a HF ``tokenizer.json``; falls back to a byte-level decoder."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer  # type: ignore

        self._tok = Tokenizer.from_file(str(path))
        self._specials: dict[str, int] = {}
        for name in (SOT, EOT, NO_TIMESTAMPS, TRANSCRIBE):
            tid = self._tok.token_to_id(name)
            if tid is not None:
                self._specials[name] = tid

    def special(self, name: str) -> int | None:
        return self._specials.get(name)

    def language_token(self, lang: str) -> int | None:
        return self._tok.token_to_id(f"<|{lang}|>")

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True).strip()

    @property
    def eot(self) -> int:
        return self._specials.get(EOT, 50257)


class WhisperAsr:
    """Encoder/decoder Whisper inference with a greedy decode loop."""

    def __init__(self, model_dir: Path, factory: SessionFactory, cfg: AsrConfig):
        self.cfg = cfg
        self.factory = factory
        self.model_dir = Path(model_dir)

        enc_path = self._find(["encoder", "WhisperEncoder"])
        dec_path = self._find(["decoder", "WhisperDecoder"])
        if enc_path is None or dec_path is None:
            raise FileNotFoundError(
                f"Whisper encoder/decoder not found under {self.model_dir}.\n"
                "Run: python scripts/download_models.py --asr"
            )

        t0 = time.perf_counter()
        self.encoder = factory.create(enc_path)
        self.decoder = factory.create(dec_path)
        log.info("Whisper loaded in %.1fs", time.perf_counter() - t0)

        self.enc_inputs = {i.name: i for i in self.encoder.get_inputs()}
        self.dec_inputs = {i.name: i for i in self.decoder.get_inputs()}
        self.dec_outputs = [o.name for o in self.decoder.get_outputs()]

        # n_mels is 80 for every Whisper except large-v3 (128). Read it off
        # the encoder's own input shape instead of guessing.
        enc_shape = next(iter(self.enc_inputs.values())).shape
        self.n_mels = 80
        for dim in enc_shape:
            if isinstance(dim, int) and dim in (80, 128):
                self.n_mels = dim
                break

        tok_path = self._find_file("tokenizer.json")
        self.tokenizer = WhisperTokenizer(tok_path) if tok_path else None
        if self.tokenizer is None:
            log.warning("tokenizer.json missing; transcripts will be token ids")

        self._config = {}
        cfg_path = self._find_file("config.json")
        if cfg_path:
            self._config = json.loads(cfg_path.read_text(encoding="utf-8"))

    # -- file discovery ----------------------------------------------------

    def _find(self, stems: list[str]) -> Path | None:
        for candidate in sorted(self.model_dir.rglob("*.onnx")):
            name = candidate.name.lower()
            if any(s.lower() in name for s in stems):
                return candidate
        return None

    def _find_file(self, name: str) -> Path | None:
        hits = list(self.model_dir.rglob(name))
        return hits[0] if hits else None

    # -- inference ---------------------------------------------------------

    def _encode(self, mel: np.ndarray) -> dict[str, np.ndarray]:
        """Run the encoder, returning whatever cross-attention state it gives."""
        name = next(iter(self.enc_inputs))
        feeds = {name: mel[None, ...].astype(np.float32)}
        outs = self.encoder.run(None, feeds)
        names = [o.name for o in self.encoder.get_outputs()]
        return dict(zip(names, outs, strict=True))

    def _initial_tokens(self, language: str | None) -> list[int]:
        if self.tokenizer is None:
            return [50258]
        tokens = [self.tokenizer.special(SOT) or 50258]
        if language:
            lang_tok = self.tokenizer.language_token(language)
            if lang_tok is not None:
                tokens.append(lang_tok)
        if (t := self.tokenizer.special(TRANSCRIBE)) is not None:
            tokens.append(t)
        if (t := self.tokenizer.special(NO_TIMESTAMPS)) is not None:
            tokens.append(t)
        return tokens

    def _decode_greedy(self, enc_state: dict[str, np.ndarray], language: str | None) -> list[int]:
        """Greedy autoregressive decode.

        Beam search would gain perhaps 1-2 WER points but multiplies decoder
        calls by the beam width, and the decoder is the part that runs once
        per token. For live captions the latency is not worth the accuracy.
        """
        tokens = self._initial_tokens(language)
        generated: list[int] = []
        eot = self.tokenizer.eot if self.tokenizer else 50257

        # The encoder's outputs feed the decoder's cross-attention inputs.
        # Match them positionally by name overlap, which is stable across
        # both the AI Hub and Optimum exports.
        cross_feeds = {
            k: v for k, v in enc_state.items() if k in self.dec_inputs
        }
        if not cross_feeds and enc_state:
            # Optimum names the encoder output "last_hidden_state" and the
            # decoder input "encoder_hidden_states".
            first = next(iter(enc_state.values()))
            for cand in ("encoder_hidden_states", "encoder_outputs", "audio_features"):
                if cand in self.dec_inputs:
                    cross_feeds = {cand: first}
                    break

        for _step in range(self.cfg.max_decode_tokens):
            feeds: dict[str, np.ndarray] = dict(cross_feeds)

            token_input = self._pick(["input_ids", "tokens", "x"])
            if token_input is None:
                raise RuntimeError(f"No token input on decoder: {list(self.dec_inputs)}")
            # AI Hub's decoder consumes one token at a time with an explicit
            # position index; Optimum's consumes the whole prefix.
            if "index" in self.dec_inputs or "position_ids" in self.dec_inputs:
                feeds[token_input] = np.array([[tokens[-1]]], dtype=np.int32)
                idx_name = "index" if "index" in self.dec_inputs else "position_ids"
                feeds[idx_name] = np.array([len(tokens) - 1], dtype=np.int32)
            else:
                feeds[token_input] = np.array([tokens], dtype=np.int64)

            # Fill any remaining required inputs with zeros of the right shape
            # (KV cache slots on the first step).
            for name, meta in self.dec_inputs.items():
                if name in feeds:
                    continue
                feeds[name] = self._zeros_for(meta)

            outs = self.decoder.run(None, feeds)
            logits = np.asarray(outs[0], dtype=np.float32)
            next_token = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())

            if next_token == eot:
                break
            tokens.append(next_token)
            generated.append(next_token)

        return generated

    def _pick(self, names: list[str]) -> str | None:
        for n in names:
            if n in self.dec_inputs:
                return n
        return None

    def _zeros_for(self, meta) -> np.ndarray:  # noqa: ANN001
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }
        dtype = dtype_map.get(meta.type, np.float32)
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
        return np.zeros(shape, dtype=dtype)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> AsrResult:
        duration = audio.size / 16_000
        t0 = time.perf_counter()

        mel = log_mel_spectrogram(audio, n_mels=self.n_mels)
        enc_state = self._encode(mel)
        token_ids = self._decode_greedy(enc_state, language or self.cfg.language)

        text = self.tokenizer.decode(token_ids) if self.tokenizer else " ".join(map(str, token_ids))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AsrResult(
            text=text,
            language=language or self.cfg.language,
            duration_s=duration,
            latency_ms=latency_ms,
            rtf=(latency_ms / 1000.0) / duration if duration > 0 else 0.0,
            provider=self.factory.provider,
            tokens=len(token_ids),
        )


# -- mock ------------------------------------------------------------------

# A real code-mixed lecture excerpt. Written out rather than generated so the
# mock demo exercises the exact thing the product claims to handle: English
# technical terms embedded in a Hindi/Tamil matrix sentence.
MOCK_TRANSCRIPT = [
    "So today we will start with eigenvalues aur eigenvectors.",
    "Matrix A ka determinant zero hoga, tabhi non-trivial solution milega.",
    "This is called the characteristic equation of the matrix.",
    "Ab hum isko diagonalize karenge using the eigenbasis.",
    "Remember, symmetric matrices always have real eigenvalues.",
    "Iska proof spectral theorem se aata hai, we will cover it next week.",
    "The rank-nullity theorem connects the null space and the column space.",
    "Practical mein, hum SVD use karte hain for numerical stability.",
]


class MockAsr:
    """Replays a scripted code-mixed transcript at realistic speed."""

    def __init__(self, cfg: AsrConfig, provider: str = "MockProvider"):
        self.cfg = cfg
        self.provider = provider
        self._i = 0

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> AsrResult:
        duration = max(audio.size / 16_000, 2.0)
        line = MOCK_TRANSCRIPT[self._i % len(MOCK_TRANSCRIPT)]
        self._i += 1
        # Pretend to take a plausible fraction of real time.
        latency_ms = 180.0 + len(line) * 1.5
        time.sleep(min(latency_ms / 1000.0, 0.4))
        return AsrResult(
            text=line,
            language="hi",
            duration_s=duration,
            latency_ms=latency_ms,
            rtf=(latency_ms / 1000.0) / duration,
            provider=self.provider,
            tokens=len(line.split()),
            meta={"mock": True},
        )


def create_asr(models_dir: Path, factory: SessionFactory, cfg: AsrConfig, mock: bool = False):
    if mock:
        return MockAsr(cfg, provider=factory.provider)
    return WhisperAsr(models_dir / cfg.model_id, factory, cfg)
