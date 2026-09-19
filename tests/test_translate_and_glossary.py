"""Code-mix handling and glossary parsing.

Term protection is the claim that distinguishes this from "call a translation
model on a caption", so it gets the most tests. A student who sees
"eigenvalue" transliterated into Devanagari cannot match it to the textbook.
"""

import pytest

from sahaay.config import GlossaryConfig
from sahaay.glossary import GlossaryWorker
from sahaay.llm import HeuristicLlm
from sahaay.translate import PassthroughTranslator, TermProtector


class TestTermProtector:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Use SVD for stability.", "SVD"),
            ("The NumPy array is fast.", "NumPy"),
            ("It takes 2.5 ms to run.", "2.5 ms"),
            ("We need 40 TOPS of compute.", "TOPS"),
        ],
    )
    def test_protects_technical_spans(self, text, expected):
        _, mapping = TermProtector().protect(text)
        assert expected in mapping.values()

    def test_round_trip_is_lossless(self):
        text = "Matrix A ka determinant zero hoga, use SVD for 2.5 ms stability."
        p = TermProtector(extra_terms=["determinant"])
        protected, mapping = p.protect(text)
        assert p.restore(protected, mapping) == text

    def test_protected_text_no_longer_contains_the_term(self):
        p = TermProtector(extra_terms=["eigenvalue"])
        protected, _ = p.protect("The eigenvalue is real.")
        assert "eigenvalue" not in protected

    def test_seeded_terms_are_protected(self):
        p = TermProtector(extra_terms=["eigenvector"])
        _, mapping = p.protect("Find the eigenvector first.")
        assert "eigenvector" in mapping.values()

    def test_ordinary_words_are_left_alone(self):
        _, mapping = TermProtector().protect("today we will start the class")
        assert mapping == {}

    def test_longest_span_wins(self):
        # "2.5 ms" must not be half-replaced by a shorter numeric match.
        p = TermProtector()
        protected, mapping = p.protect("It took 2.5 ms exactly.")
        assert p.restore(protected, mapping) == "It took 2.5 ms exactly."

    def test_repeated_term_restores_everywhere(self):
        p = TermProtector(extra_terms=["gradient"])
        text = "The gradient is zero when the gradient vanishes."
        protected, mapping = p.protect(text)
        assert p.restore(protected, mapping) == text


class TestPassthrough:
    def test_flags_itself_so_the_ui_can_say_so(self):
        # Showing English text under a Hindi heading without saying so would
        # be worse than showing nothing.
        result = PassthroughTranslator().translate("hello", "hi")
        assert result.passthrough is True
        assert result.text == "hello"


class TestHeuristicGlossary:
    def test_finds_seeded_terms(self):
        out = HeuristicLlm().generate("<transcript>Today: eigenvalue and entropy.</transcript>")
        assert "eigenvalue" in out.text

    @pytest.mark.parametrize("word", ["solution", "equation", "function", "important"])
    def test_ignores_ordinary_academic_words(self, word):
        # These pass a naive -tion/-ity filter and would fill the sidebar
        # with noise, which reads as a broken product.
        out = HeuristicLlm().generate(f"<transcript>The {word} is here.</transcript>")
        assert word not in out.text

    def test_caps_output(self):
        text = "eigenvalue determinant entropy gradient convolution mitosis"
        out = HeuristicLlm().generate(f"<transcript>{text}</transcript>")
        assert len(out.text.splitlines()) <= 3

    def test_reports_its_backend(self):
        assert HeuristicLlm().generate("<transcript>eigenvalue</transcript>").backend == "heuristic"


class TestGlossaryParsing:
    def worker(self) -> GlossaryWorker:
        return GlossaryWorker(HeuristicLlm(), GlossaryConfig(), "hi")

    def test_parses_the_expected_format(self):
        out = self.worker()._parse("entropy :: kitna bikhra hua hai", "src")
        assert len(out) == 1
        assert out[0].term == "entropy"

    def test_strips_model_decoration(self):
        # Models add numbering and bold whatever the prompt says.
        out = self.worker()._parse("1. **entropy** :: disorder", "src")
        assert out[0].term == "entropy"

    def test_none_yields_nothing(self):
        assert self.worker()._parse("NONE", "src") == []

    def test_lines_without_separator_are_skipped(self):
        assert self.worker()._parse("Sure, here are the terms!", "src") == []

    def test_rejects_absurdly_long_terms(self):
        # A run-on line means the model ignored the format; better to drop it
        # than to render a paragraph as a glossary heading.
        long_term = "x" * 100
        assert self.worker()._parse(f"{long_term} :: meaning", "src") == []

    def test_deduplicates_within_a_session(self):
        w = self.worker()
        w._process(["eigenvalue is here"])
        first = len(w.entries)
        w._process(["eigenvalue again"])
        assert len(w.entries) == first
