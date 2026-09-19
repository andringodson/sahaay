"""Whisper language handling.

The decoder prompt on a multilingual model is
``[SOT, <|lang|>, <|transcribe|>, <|notimestamps|>]`` and the language token
is **not optional**. Omitting it puts the decoder out of distribution and it
degenerates: measured on whisper-small with a 15 s clip, the output was
"The is the................" for 220 tokens.

That mattered because ``AsrConfig.language`` defaults to None, so the
shipped default produced exactly that on any multilingual model. English-only
models were unaffected, which is why it survived earlier testing.
"""

import numpy as np
import pytest

from sahaay.asr import WHISPER_LANGUAGES


class TestLanguageTable:
    def test_covers_the_eight_target_languages(self):
        from sahaay.config import SUPPORTED_LANGUAGES

        for code in SUPPORTED_LANGUAGES:
            assert code in WHISPER_LANGUAGES, f"{code} is a target language but not detectable"

    def test_indian_languages_present(self):
        for code in ("hi", "ta", "te", "kn", "ml", "bn", "mr", "gu"):
            assert code in WHISPER_LANGUAGES

    def test_no_duplicates(self):
        assert len(WHISPER_LANGUAGES) == len(set(WHISPER_LANGUAGES))


class FakeTokenizer:
    """Minimal stand-in: only language tokens matter for detection."""

    def __init__(self, languages: dict[str, int], sot: int = 50258):
        self._languages = languages
        self._sot = sot

    def special(self, name):  # noqa: ANN001
        return self._sot if name.startswith("<|startoftranscript") else None

    def language_token(self, lang):  # noqa: ANN001
        return self._languages.get(lang)


class FakeAsr:
    """Exercises detect_language without loading a 300 MB graph."""

    from sahaay.asr import WhisperAsr

    detect_language = WhisperAsr.detect_language

    def __init__(self, languages: dict[str, int], winner: int | None, vocab: int = 52000):
        self.tokenizer = FakeTokenizer(languages)
        self._language_tokens = languages
        self._winner = winner
        self._vocab = vocab
        self.raised = False

    def _decode_step(self, tokens, enc_state):  # noqa: ANN001
        if self._winner is None:
            self.raised = True
            raise RuntimeError("decoder exploded")
        logits = np.zeros((1, 1, self._vocab), dtype=np.float32)
        logits[0, 0, self._winner] = 10.0
        return logits


class TestDetection:
    LANGS = {"en": 50259, "hi": 50276, "ta": 50287}

    def test_picks_the_highest_scoring_language(self):
        asr = FakeAsr(self.LANGS, winner=50276)
        assert asr.detect_language({}) == "hi"

    def test_ignores_non_language_tokens(self):
        # An ordinary word may well outscore every language tag; the argmax
        # is restricted to language tokens so it can never win.
        asr = FakeAsr(self.LANGS, winner=99)
        assert asr.detect_language({}) in self.LANGS

    def test_english_only_model_returns_none(self):
        # whisper-*.en has no language tokens at all.
        assert FakeAsr({}, winner=50259).detect_language({}) is None

    def test_decoder_failure_is_not_fatal(self):
        # Detection is a nicety; losing it must not lose the caption.
        asr = FakeAsr(self.LANGS, winner=None)
        assert asr.detect_language({}) is None
        assert asr.raised

    def test_out_of_range_token_is_handled(self):
        # A tokenizer/graph mismatch must not raise IndexError mid-lecture.
        asr = FakeAsr({"en": 999_999}, winner=1, vocab=100)
        assert asr.detect_language({}) is None


class TestInitialTokens:
    def build(self, languages):
        from sahaay.asr import WhisperAsr

        obj = object.__new__(WhisperAsr)
        obj.tokenizer = FakeTokenizer(languages)
        obj._language_tokens = languages
        return obj

    def test_language_token_is_included_when_known(self):
        from sahaay.asr import WhisperAsr

        obj = self.build({"hi": 50276})
        # _initial_tokens also asks for transcribe/notimestamps, which the
        # fake tokenizer reports as absent - the language token is the part
        # under test.
        tokens = WhisperAsr._initial_tokens(obj, "hi")
        assert 50276 in tokens

    def test_unknown_language_is_skipped_not_crashed(self):
        from sahaay.asr import WhisperAsr

        obj = self.build({"hi": 50276})
        assert WhisperAsr._initial_tokens(obj, "xx")  # returns SOT at minimum


@pytest.mark.parametrize("code", ["hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "en"])
def test_every_ui_language_is_a_whisper_language(code):
    assert code in WHISPER_LANGUAGES
