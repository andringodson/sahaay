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

from .config import SUPPORTED_LANGUAGES, TranslateConfig, resolve_model_id
from .kvcache import CACHE_FLAG, MergedDecoderCache
from .llm import SEED_GLOSSARY
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

# Units must be an explicit list, not "any short token". The original
# pattern allowed [a-zA-Z]{1,4} after a number, which swallowed the next
# ordinary word: "must be 0 for a solution" protected the span "0 for",
# and the Hindi translation came back mangled around the hole it left.
_UNITS = (
    "ms|us|ns|s|min|h|Hz|kHz|MHz|GHz|"
    "mm|cm|m|km|nm|um|"
    "mg|g|kg|"
    "V|mV|A|mA|W|kW|J|kJ|N|Pa|kPa|"
    "K|C|F|"
    "b|B|KB|MB|GB|TB|bit|bits|byte|bytes|"
    "TOPS|FLOPS|px|dB|deg|rad"
)
_NUMERIC = re.compile(rf"\b\d+(?:\.\d+)?(?:\s?(?:{_UNITS})|%)?\b")


class TermProtector:
    """Swap protected spans for placeholders around the translation call.

    Placeholders are chosen to survive subword tokenisation: a bare token
    like ``__T0__`` gets split and mangled, whereas a short alphanumeric
    sentinel in the model's vocabulary usually round-trips intact.
    """

    SENTINEL = "Qx{}z"

    def __init__(self, extra_terms: list[str] | None = None):
        self.extra = {t.lower() for t in (extra_terms or [])}

    def _is_protected(self, word: str) -> bool:
        """Match the seeded vocabulary allowing for simple inflection.

        Lectures say "eigenvalues", the glossary stores "eigenvalue". Exact
        matching missed the plural and let it be translated - observed as
        Telugu rendering "eigenvalues" as "self values". Strip the common
        English endings rather than pulling in a stemmer for eight suffixes.
        """
        w = word.lower().strip("-")
        if w in self.extra:
            return True
        for suffix in ("s", "es", "ed", "ing"):
            if not w.endswith(suffix):
                continue
            stem = w[: -len(suffix)]
            # "diagonalizing" -> "diagonaliz" -> "diagonalize": English drops
            # the silent e before -ing/-ed, so put it back before matching.
            if stem in self.extra or stem + "e" in self.extra:
                return True
        # "matrices" -> "matrix", "indices" -> "index"
        return w.endswith("ices") and w[:-4] + "ix" in self.extra

    def protect(self, text: str) -> tuple[str, dict[str, str]]:
        mapping: dict[str, str] = {}
        spans: list[str] = []

        for pattern in (_ACRONYM, _CAMEL, _NUMERIC):
            spans.extend(m.group(0) for m in pattern.finditer(text))
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text):
            if self._is_protected(word):
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
        self.dec_outputs = [o.name for o in self.decoder.get_outputs()]
        self._cache = MergedDecoderCache(self.dec_inputs, self.dec_outputs)

        tok_files = list(self.model_dir.rglob("tokenizer.json"))
        if not tok_files:
            raise FileNotFoundError(f"tokenizer.json missing under {model_dir}")
        self.tokenizer = Tokenizer.from_file(str(tok_files[0]))
        # Seed the protector with the STEM vocabulary. Without this it only
        # catches acronyms, CamelCase and numerics - so "SVD" survived but
        # "eigenvalue" was cheerfully translated into Telugu as "self
        # values", which is precisely the failure this module exists to
        # prevent. Measured, not hypothetical.
        self.protector = TermProtector(extra_terms=list(SEED_GLOSSARY))

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

        # NLLB generation starts from EOS, then the forced target-language
        # token. Getting this prefix wrong produces fluent output in the
        # wrong language, which is worse than an error because it looks fine.
        eos = self.tokenizer.token_to_id("</s>")
        eos = 2 if eos is None else eos
        decoded: list[int] = [eos, self._lang_token(lang["nllb"])]

        past: dict[str, np.ndarray] = {}

        for step in range(self.cfg.max_tokens):
            first_pass = step == 0
            feeds: dict[str, np.ndarray] = {
                "encoder_hidden_states": enc_out,
                "encoder_attention_mask": attention,
            }
            feeds = {k: v for k, v in feeds.items() if k in self.dec_inputs}

            if self._cache.active:
                if first_pass:
                    feeds["input_ids"] = np.array([decoded], dtype=np.int64)
                    feeds.update(self._cache.empty())
                else:
                    feeds["input_ids"] = np.array([[decoded[-1]]], dtype=np.int64)
                    feeds.update(past)
                feeds[CACHE_FLAG] = MergedDecoderCache.flag(first_pass)
            else:
                feeds["input_ids"] = np.array([decoded], dtype=np.int64)

            outs = self.decoder.run(None, feeds)
            logits = np.asarray(outs[0], dtype=np.float32)
            nxt = int(logits[0, -1].argmax())

            if self._cache.active:
                past = self._cache.collect(outs, reuse_encoder=not first_pass, previous=past)

            if nxt == eos:
                break
            decoded.append(nxt)

        out = self.tokenizer.decode(decoded[2:], skip_special_tokens=True).strip()
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
        model_id = resolve_model_id(models_dir, cfg.model_id, cfg.candidates)
        return NllbTranslator(models_dir / model_id, factory, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("translator unavailable (%s); captions will not be translated", exc)
        return PassthroughTranslator(provider=factory.provider)
