"""Caption translation into Indian languages.

NLLB-200-distilled-600M, INT8-quantised. 600M is the sweet spot: it covers
all eight target languages in one model, and at INT8 it is small enough to
sit on the NPU next to Whisper and the 3B LLM without thrashing.

A note on what this stage is *not* doing. Indian lecture speech is heavily
code-mixed - English technical terms inside a Hindi or Tamil sentence. Naive
translation destroys exactly the terms the student needs to recognise in the
textbook and the exam. So technical tokens are protected before translation
and restored afterwards, which is why a translated caption still reads
"eigenvalue" and not a transliterated guess at it.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SUPPORTED_LANGUAGES, TranslateConfig
from .runtime import SessionFactory

log = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    text: str
    target_language: str
    latency_ms: float
    provider: str
    protected_terms: list[str]
    passthrough: bool = False


# Terms we refuse to translate: acronyms, units, anything already in the
# seeded technical vocabulary, and CamelCase identifiers.
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_CAMEL = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
_NUMERIC = re.compile(r"\b\d+(?:\.\d+)?(?:\s?[a-zA-Z%]{1,4})?\b")


class TermProtector:
    """Swap protected spans for placeholders around the translation call.

    Placeholders are chosen to survive subword tokenisation: a bare token
    like ``__T0__`` gets split and mangled, whereas a short alphanumeric
    sentinel in the model's vocabulary usually round-trips intact.
    """

    SENTINEL = "Qx{}z"

    def __init__(self, extra_terms: list[str] | None = None):
        self.extra = {t.lower() for t in (extra_terms or [])}

    def protect(self, text: str) -> tuple[str, dict[str, str]]:
        mapping: dict[str, str] = {}
        spans: list[str] = []

        for pattern in (_ACRONYM, _CAMEL, _NUMERIC):
            spans.extend(m.group(0) for m in pattern.finditer(text))
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text):
            if word.lower() in self.extra:
                spans.append(word)

        out = text
        # Longest first, so "SVD matrix" doesn't get half-replaced.
        for i, span in enumerate(sorted(set(spans), key=len, reverse=True)):
            token = self.SENTINEL.format(i)
            mapping[token] = span
            out = re.sub(rf"(?<!\w){re.escape(span)}(?!\w)", token, out)
        return out, mapping

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        for token, original in mapping.items():
            # Models sometimes alter the sentinel's case or spacing.
            text = re.sub(re.escape(token), original, text, flags=re.I)
            text = re.sub(re.escape(token.replace("z", " z")), original, text, flags=re.I)
        return text


class NllbTranslator:
    """Encoder/decoder NLLB inference with greedy decoding."""

    def __init__(self, model_dir: Path, factory: SessionFactory, cfg: TranslateConfig):
        from tokenizers import Tokenizer  # type: ignore

        self.cfg = cfg
        self.factory = factory
        self.model_dir = Path(model_dir)

        enc = self._find("encoder")
        dec = self._find("decoder")
        if enc is None or dec is None:
            raise FileNotFoundError(
                f"NLLB encoder/decoder not found under {model_dir}.\n"
                "Run: python scripts/download_models.py --translate"
            )

        self.encoder = factory.create(enc)
        self.decoder = factory.create(dec)
        self.dec_inputs = {i.name: i for i in self.decoder.get_inputs()}

        tok_files = list(self.model_dir.rglob("tokenizer.json"))
        if not tok_files:
            raise FileNotFoundError(f"tokenizer.json missing under {model_dir}")
        self.tokenizer = Tokenizer.from_file(str(tok_files[0]))
        self.protector = TermProtector()

    def _find(self, stem: str) -> Path | None:
        for p in sorted(self.model_dir.rglob("*.onnx")):
            if stem in p.name.lower():
                return p
        return None

    def _lang_token(self, lang_code: str) -> int:
        tid = self.tokenizer.token_to_id(lang_code)
        if tid is None:
            raise ValueError(f"NLLB language token not in vocabulary: {lang_code}")
        return tid

    def translate(self, text: str, target_language: str) -> TranslationResult:
        t0 = time.perf_counter()
        lang = SUPPORTED_LANGUAGES.get(target_language)
        if lang is None:
            raise ValueError(f"Unsupported target language: {target_language}")

        protected, mapping = self.protector.protect(text)
        enc_ids = self.tokenizer.encode(protected).ids
        input_ids = np.array([enc_ids], dtype=np.int64)
        attention = np.ones_like(input_ids)

        enc_out = self.encoder.run(
            None, {"input_ids": input_ids, "attention_mask": attention}
        )[0]

        # NLLB forces the target language as the first decoder token.
        decoded: list[int] = [self._lang_token(lang["nllb"])]
        eos = self.tokenizer.token_to_id("</s>") or 2

        for _ in range(self.cfg.max_tokens):
            feeds = {
                "encoder_hidden_states": enc_out,
                "encoder_attention_mask": attention,
                "input_ids": np.array([decoded], dtype=np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in self.dec_inputs}
            if "use_cache_branch" in self.dec_inputs:
                feeds["use_cache_branch"] = np.array([False])

            logits = self.decoder.run(None, feeds)[0]
            nxt = int(np.asarray(logits)[0, -1].argmax())
            if nxt == eos:
                break
            decoded.append(nxt)

        out = self.tokenizer.decode(decoded[1:], skip_special_tokens=True).strip()
        out = self.protector.restore(out, mapping)

        return TranslationResult(
            text=out,
            target_language=target_language,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            provider=self.factory.provider,
            protected_terms=list(mapping.values()),
        )


class PassthroughTranslator:
    """Used when NLLB weights are absent.

    Returns the source text unchanged and flags it, so the UI can say
    "translation model not installed" instead of silently showing English
    captions in a Hindi-labelled pane.
    """

    def __init__(self, provider: str = "none"):
        self.provider = provider
        self.protector = TermProtector()

    def translate(self, text: str, target_language: str) -> TranslationResult:
        t0 = time.perf_counter()
        _, mapping = self.protector.protect(text)
        return TranslationResult(
            text=text,
            target_language=target_language,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            provider=self.provider,
            protected_terms=list(mapping.values()),
            passthrough=True,
        )


def create_translator(
    models_dir: Path, factory: SessionFactory, cfg: TranslateConfig, mock: bool = False
):
    if not cfg.enabled or mock:
        return PassthroughTranslator(provider=factory.provider if not mock else "mock")
    try:
        return NllbTranslator(models_dir / cfg.model_id, factory, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("translator unavailable (%s); captions will not be translated", exc)
        return PassthroughTranslator(provider=factory.provider)
