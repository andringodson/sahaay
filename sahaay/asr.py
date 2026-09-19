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

from .config import AsrConfig, resolve_model_id
from .features import log_mel_spectrogram
from .kvcache import CACHE_FLAG, MergedDecoderCache
from .runtime import SessionFactory

log = logging.getLogger(__name__)

# Whisper special tokens (multilingual vocabulary).
SOT = "<|startoftranscript|>"
EOT = "<|endoftext|>"
NO_TIMESTAMPS = "<|notimestamps|>"
TRANSCRIBE = "<|transcribe|>"

# Whisper's language codes. The decoder prompt is
# [SOT, <|lang|>, <|transcribe|>, <|notimestamps|>] and the language token is
# NOT optional on a multilingual model: leaving it out puts the decoder
# out of distribution and it degenerates. Measured on whisper-small with a
# 15 s clip, omitting it produced "The is the................" for 220
# tokens. So when no language is configured we detect one.
#
# Indian languages first purely for readability; order has no effect.
WHISPER_LANGUAGES = (
    "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "ur", "ne", "si",
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "vi", "he", "uk", "el", "ms", "cs", "ro",
    "da", "hu", "fi", "no", "th", "sk", "fa", "sw", "bg", "hr", "lt", "az",
)


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

        enc_path = self._find(["encoder_model", "encoder", "WhisperEncoder"])
        # The merged decoder carries both the cache and no-cache branches;
        # `decoder_with_past_model` is the cache-only half and cannot run the
        # first step, so it is excluded rather than ranked lower.
        dec_path = self._find(
            ["decoder_model_merged", "decoder_model", "decoder", "WhisperDecoder"],
            avoid=("with_past",),
        )
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

        # Decide the packaging once, from the graph itself, rather than on
        # every decode step. See the module docstring.
        self._index_name = next(
            (n for n in ("index", "position_ids") if n in self.dec_inputs), None
        )
        self._positional = self._index_name is not None
        self._cache = MergedDecoderCache(self.dec_inputs, self.dec_outputs)
        self._merged = not self._positional and self._cache.active
        log.info(
            "decoder packaging: %s",
            "AI Hub (positional index)" if self._positional
            else "Optimum merged (KV cache)" if self._merged
            else "plain (no cache)",
        )

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

        # An English-only model (whisper-*.en) has no language tokens at all,
        # which is how we tell the two apart without reading config files.
        self._language_tokens: dict[str, int] = {}
        if self.tokenizer is not None:
            for code in WHISPER_LANGUAGES:
                tid = self.tokenizer.language_token(code)
                if tid is not None:
                    self._language_tokens[code] = tid
        self.multilingual = bool(self._language_tokens)
        log.info(
            "whisper is %s",
            f"multilingual ({len(self._language_tokens)} languages)"
            if self.multilingual else "English-only",
        )

        self._config = {}
        cfg_path = self._find_file("config.json")
        if cfg_path:
            self._config = json.loads(cfg_path.read_text(encoding="utf-8"))

    # -- file discovery ----------------------------------------------------

    # Variant suffixes shipped alongside the full-precision graph. We do not
    # want these picked by accident: the model directory may legitimately
    # hold a dozen of them, and quantisation choice belongs in the download
    # step, not in a glob that happens to sort a certain way.
    _VARIANTS = ("int8", "uint8", "fp16", "q4", "q4f16", "bnb4", "quantized")

    def _find(self, stems: list[str], avoid: tuple[str, ...] = ()) -> Path | None:
        """Pick one graph out of a directory that may hold many variants.

        Preference: an exact stem match at full precision, then any
        full-precision match, then a quantised one. Without this, a Whisper
        repo with 24 .onnx files would hand back `decoder_model_bnb4.onnx`
        or `decoder_with_past_model.onnx` purely on sort order.
        """
        candidates = sorted(self.model_dir.rglob("*.onnx"))
        if not candidates:
            return None

        def matches(p: Path) -> bool:
            name = p.stem.lower()
            if any(a in name for a in avoid):
                return False
            return any(s.lower() in name for s in stems)

        pool = [p for p in candidates if matches(p)]
        if not pool:
            return None

        full = [p for p in pool if not any(v in p.stem.lower() for v in self._VARIANTS)]
        preferred = full or pool

        for stem in stems:
            for p in preferred:
                if p.stem.lower() == stem.lower():
                    return p
        return preferred[0]

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

    def detect_language(self, enc_state: dict[str, np.ndarray]) -> str | None:
        """Run one decode step from SOT and read off the language.

        This is how Whisper itself does it: with only the start token in the
        prompt, the highest-probability next token is the language tag. We
        restrict the argmax to language tokens so an ordinary word can never
        win, then cache nothing - a code-mixed lecture genuinely changes
        language between segments, which is the whole point of detecting per
        segment rather than once per session.
        """
        if self.tokenizer is None or not self._language_tokens:
            return None

        sot = self.tokenizer.special(SOT)
        if sot is None:
            return None

        try:
            logits = self._decode_step([sot], enc_state)
        except Exception as exc:  # noqa: BLE001 - detection must never be fatal
            log.debug("language detection failed (%s); falling back to English", exc)
            return None

        ids = np.array(list(self._language_tokens.values()), dtype=np.int64)
        row = logits.reshape(-1, logits.shape[-1])[-1]
        if ids.max() >= row.shape[0]:
            return None
        best = int(ids[int(np.argmax(row[ids]))])
        codes = {v: k for k, v in self._language_tokens.items()}
        return codes.get(best)

    def _decode_step(self, tokens: list[int], enc_state: dict[str, np.ndarray]) -> np.ndarray:
        """One decoder pass over ``tokens``, returning raw logits."""
        cross_feeds = self._cross_feeds(enc_state)
        token_input = self._pick(["input_ids", "tokens", "x"])
        if token_input is None:
            raise RuntimeError("decoder has no token input")

        feeds: dict[str, np.ndarray] = dict(cross_feeds)
        if self._positional:
            feeds[token_input] = np.array([[tokens[-1]]], dtype=np.int32)
            feeds[self._index_name] = np.array([len(tokens) - 1], dtype=np.int32)
        else:
            feeds[token_input] = np.array([tokens], dtype=np.int64)
            if self._merged:
                feeds.update(self._cache.empty())
                feeds[CACHE_FLAG] = MergedDecoderCache.flag(True)

        for name, meta in self.dec_inputs.items():
            if name not in feeds:
                feeds[name] = self._zeros_for(meta)

        return np.asarray(self.decoder.run(None, feeds)[0], dtype=np.float32)

    def _cross_feeds(self, enc_state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feeds = {k: v for k, v in enc_state.items() if k in self.dec_inputs}
        if not feeds and enc_state:
            first = next(iter(enc_state.values()))
            for cand in ("encoder_hidden_states", "encoder_outputs", "audio_features"):
                if cand in self.dec_inputs:
                    return {cand: first}
        return feeds

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
        cross_feeds = self._cross_feeds(enc_state)

        token_input = self._pick(["input_ids", "tokens", "x"])
        if token_input is None:
            raise RuntimeError(f"No token input on decoder: {list(self.dec_inputs)}")

        past: dict[str, np.ndarray] = {}

        for step in range(self.cfg.max_decode_tokens):
            feeds: dict[str, np.ndarray] = dict(cross_feeds)
            first_pass = step == 0

            if self._positional:
                # AI Hub: one token at a time plus an explicit position.
                feeds[token_input] = np.array([[tokens[-1]]], dtype=np.int32)
                feeds[self._index_name] = np.array([len(tokens) - 1], dtype=np.int32)
            elif self._merged:
                # Optimum merged decoder. The first pass computes the caches
                # from the whole prompt; every pass after feeds back the
                # previous step's `present.*` and submits only the newest
                # token. Re-sending the full prefix each step also "works",
                # but makes decoding quadratic - on a 200-token caption that
                # is the difference between live and not.
                if first_pass:
                    feeds[token_input] = np.array([tokens], dtype=np.int64)
                    feeds.update(self._cache.empty())
                else:
                    feeds[token_input] = np.array([[tokens[-1]]], dtype=np.int64)
                    feeds.update(past)
                feeds[CACHE_FLAG] = MergedDecoderCache.flag(first_pass)
            else:
                feeds[token_input] = np.array([tokens], dtype=np.int64)

            # Anything still unfilled gets a correctly-typed zero tensor.
            for name, meta in self.dec_inputs.items():
                if name not in feeds:
                    feeds[name] = self._zeros_for(meta)

            outs = self.decoder.run(None, feeds)
            logits = np.asarray(outs[0], dtype=np.float32)
            next_token = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())

            if self._merged:
                past = self._cache.collect(outs, reuse_encoder=not first_pass, previous=past)

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
            # Missing bool was a real failure: use_cache_branch is bool, and
            # feeding it float32 made ORT reject the whole decoder call.
            "tensor(bool)": np.bool_,
        }
        dtype = dtype_map.get(meta.type, np.float32)
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
        return np.zeros(shape, dtype=dtype)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> AsrResult:
        duration = audio.size / 16_000
        t0 = time.perf_counter()

        mel = log_mel_spectrogram(audio, n_mels=self.n_mels)
        enc_state = self._encode(mel)

        chosen = language or self.cfg.language
        detected = None
        if chosen is None and self.multilingual:
            detected = self.detect_language(enc_state)
            # English is the safe default: it keeps the English technical
            # terms intact, which is what the student needs to recognise.
            chosen = detected or "en"

        token_ids = self._decode_greedy(enc_state, chosen)

        text = self.tokenizer.decode(token_ids) if self.tokenizer else " ".join(map(str, token_ids))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AsrResult(
            text=text,
            language=chosen,
            duration_s=duration,
            latency_ms=latency_ms,
            rtf=(latency_ms / 1000.0) / duration if duration > 0 else 0.0,
            provider=self.factory.provider,
            tokens=len(token_ids),
            meta={"detected_language": detected} if detected else {},
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
    model_id = resolve_model_id(models_dir, cfg.model_id, cfg.candidates)
    log.info("ASR model: %s", model_id)
    return WhisperAsr(models_dir / model_id, factory, cfg)
