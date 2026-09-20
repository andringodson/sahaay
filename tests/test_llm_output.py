"""Parsing and cleanup of real language-model output.

Every case here came out of an actual Llama 3.2 run. The model produced
**four different shapes** for the same quiz prompt across four runs, which is
why the parser is structural rather than format-matching: a parser keyed to
the format the prompt asked for returned zero items from output that was
otherwise perfectly good.
"""

import pytest

from sahaay.llm import find_genai_model
from sahaay.notes import NotesWriter


class TestQuizShapes:
    """The four shapes observed, plus what must be rejected."""

    @pytest.mark.parametrize(
        "name,text,expected",
        [
            ("as prompted", "Q :: What is entropy? :: A measure of disorder.", 1),
            ("answer on next line", "Q :: What is entropy?\nA measure of disorder.", 1),
            (
                "no markers at all",
                "What is entropy?\nA measure of disorder.\n\n"
                "What is a matrix?\nA grid of numbers.",
                2,
            ),
            (
                # This one broke block-based parsing: a blank line between the
                # question and its answer put them in separate blocks.
                "numbered, blank line between Q and A",
                "Here are five questions:\n\n1. What is entropy?\n\n"
                "A measure of disorder.\n\n2. What is a matrix?\n\nA grid of numbers.",
                2,
            ),
            ("labelled answer", "1. What is entropy?\nA: A measure of disorder.", 1),
        ],
    )
    def test_shapes(self, name, text, expected):
        assert len(NotesWriter._parse_quiz(text)) == expected, name

    def test_preamble_is_not_a_question(self):
        items = NotesWriter._parse_quiz(
            "Here are five short self-test questions:\n\nWhat is entropy?\nDisorder."
        )
        assert len(items) == 1
        assert items[0].question == "What is entropy?"

    def test_prose_is_rejected(self):
        assert NotesWriter._parse_quiz("The lecture covered entropy.\nIt was good.") == []

    def test_multiple_choice_is_dropped(self):
        # An "answer" that is a list of options leaves the card with no
        # answer, which is worse than omitting the question.
        assert NotesWriter._parse_quiz("1. What is entropy?\nA) disorder B) order C) heat") == []

    def test_answer_label_is_stripped(self):
        items = NotesWriter._parse_quiz("What is entropy?\nAnswer: A measure of disorder.")
        assert items[0].answer == "A measure of disorder."

    def test_multiline_answers_are_joined(self):
        items = NotesWriter._parse_quiz("What is entropy?\nA measure of\ndisorder in a system.")
        assert items[0].answer == "A measure of disorder in a system."


class TestClean:
    def test_unicode_bullets_become_markdown(self):
        # Models write "•"; the UI renderer only understands "- ", so the
        # notes rendered as one run-on paragraph.
        out = NotesWriter._clean("## Topics\n• Eigenvalues\n• Determinants")
        assert "- Eigenvalues" in out
        assert "•" not in out

    def test_truncated_trailing_bullet_is_dropped(self):
        # Hitting the token budget mid-word reads as a bug, not a limit.
        out = NotesWriter._clean(
            "## Key points\n- Symmetric matrices have real eigenvalues.\n- Real eigenvalues are "
        )
        assert "Real eigenvalues are" not in out
        assert "Symmetric matrices" in out

    def test_complete_trailing_bullet_is_kept(self):
        out = NotesWriter._clean("## Key points\n- Determinant of A is zero.")
        assert "Determinant of A is zero." in out

    def test_assistant_preamble_is_stripped(self):
        assert not NotesWriter._clean("Sure, here are your notes!\n## Topics").startswith("Sure")


class TestGenAiDiscovery:
    def test_finds_nested_config(self, tmp_path):
        # Published GenAI builds nest by target; pointing at the download
        # root fails, and the glossary silently drops to heuristics.
        nested = tmp_path / "llama" / "cpu_and_mobile" / "cpu-int4-rtn-block-32"
        nested.mkdir(parents=True)
        (nested / "genai_config.json").write_text("{}", encoding="utf-8")
        assert find_genai_model(tmp_path / "llama") == nested

    def test_prefers_cpu_over_cuda(self, tmp_path):
        root = tmp_path / "llama"
        for variant in ("cuda/cuda-int4", "cpu_and_mobile/cpu-int4"):
            d = root / variant
            d.mkdir(parents=True)
            (d / "genai_config.json").write_text("{}", encoding="utf-8")
        # A CUDA build would never load on a Snapdragon laptop. Compare the
        # path relative to root - tmp_path itself contains the test name,
        # which contains "cuda".
        chosen = find_genai_model(root).relative_to(root)
        assert "cuda" not in str(chosen)
        assert "cpu" in str(chosen)

    def test_accepts_a_flat_layout(self, tmp_path):
        (tmp_path / "genai_config.json").write_text("{}", encoding="utf-8")
        assert find_genai_model(tmp_path) == tmp_path

    def test_missing_returns_none(self, tmp_path):
        assert find_genai_model(tmp_path / "nope") is None


class TestGlossarySeparators:
    """The prompt asks for "::". The model does not always oblige.

    Adding a repetition penalty - needed to stop small models looping -
    makes them avoid the repeated "::" token and switch to "-". A parser
    keyed to "::" turned a batch of three good Hindi glosses into zero
    entries, silently.
    """

    def worker(self):
        from sahaay.config import GlossaryConfig
        from sahaay.glossary import GlossaryWorker
        from sahaay.llm import HeuristicLlm

        return GlossaryWorker(HeuristicLlm(), GlossaryConfig(), "hi")

    @pytest.mark.parametrize(
        "line,term",
        [
            ("entropy :: a measure of disorder", "entropy"),
            ("1. Eigenvalue - एक मैट्रिक्स का निर्धारक तत्व", "Eigenvalue"),
            ("2. Eigenvector – a special direction", "Eigenvector"),
            ("3. Determinant: मैट्रिक्स का एक मान", "Determinant"),
        ],
    )
    def test_separators(self, line, term):
        out = self.worker()._parse(line, "src")
        assert len(out) == 1
        assert out[0].term == term

    def test_hyphenated_words_are_not_split(self):
        # "non-trivial" must not look like "non" :: "trivial ...". The dash
        # separator requires surrounding spaces precisely for this.
        assert self.worker()._parse("non-trivial solution ka matlab", "src") == []

    def test_preamble_is_ignored(self):
        # Models open with a sentence ending in a colon.
        assert self.worker()._parse("यहाँ 3 तकनीकी शब्द हैं:", "src") == []

    def test_truncated_gloss_is_dropped(self):
        assert self.worker()._parse("Characteristic Equation :: मैट्र", "src") == []


class TestModelResolution:
    """Model ids resolve against what is on disk.

    Pinning a name was a real deployment bug: the default was the AI Hub
    model, but `download_models.py --auto` on x86 fetches the portable one,
    so a clean install found no ASR at all.
    """

    def setup_models(self, root, *names):
        for name in names:
            d = root / name / "onnx"
            d.mkdir(parents=True)
            (d / "encoder_model.onnx").write_bytes(b"\0")
        return root

    def test_prefers_snapdragon_build(self, tmp_path):
        from sahaay.config import AsrConfig, resolve_model_id

        self.setup_models(tmp_path, "whisper_small_portable", "whisper_small_quantized")
        cfg = AsrConfig()
        assert resolve_model_id(tmp_path, cfg.model_id, cfg.candidates) == "whisper_small_quantized"

    def test_falls_back_to_portable(self, tmp_path):
        from sahaay.config import AsrConfig, resolve_model_id

        self.setup_models(tmp_path, "whisper_small_portable")
        cfg = AsrConfig()
        assert resolve_model_id(tmp_path, cfg.model_id, cfg.candidates) == "whisper_small_portable"

    def test_explicit_name_is_honoured(self, tmp_path):
        from sahaay.config import AsrConfig, resolve_model_id

        self.setup_models(tmp_path, "whisper_small_quantized")
        cfg = AsrConfig()
        assert resolve_model_id(tmp_path, "whisper_tiny_en", cfg.candidates) == "whisper_tiny_en"

    def test_empty_directory_is_not_selected(self, tmp_path):
        # A directory with no .onnx is a half-finished download, not a model.
        from sahaay.config import AsrConfig, resolve_model_id

        (tmp_path / "whisper_small_quantized").mkdir(parents=True)
        self.setup_models(tmp_path, "whisper_small_portable")
        cfg = AsrConfig()
        assert resolve_model_id(tmp_path, cfg.model_id, cfg.candidates) == "whisper_small_portable"

    def test_nothing_present_names_the_preferred_model(self, tmp_path):
        # So the error message points at what the user most likely wanted.
        from sahaay.config import AsrConfig, resolve_model_id

        cfg = AsrConfig()
        assert resolve_model_id(tmp_path, cfg.model_id, cfg.candidates) == cfg.candidates[0]


class TestDeviceAwareModelChoice:
    """Which glossary model to prefer depends on the hardware.

    Measured on an x86 CPU: the 3B runs at 7.4 tok/s against the 1B's 16.9 -
    the expected ratio - but it needs ~3.5 GB resident, and under memory
    pressure one 146-token glossary call was observed taking 35 minutes.
    A glossary entry that arrives after the lecture has ended is not a
    glossary entry, so on CPU the smaller model wins.
    """

    def candidates(self) -> list[str]:
        from sahaay.config import GlossaryConfig

        return GlossaryConfig().candidates

    def test_npu_keeps_the_configured_order(self):
        from sahaay.llm import order_candidates

        c = self.candidates()
        assert order_candidates(c, npu_active=True) == c

    def test_npu_prefers_the_larger_model(self):
        from sahaay.llm import order_candidates

        assert "3b" in order_candidates(self.candidates(), npu_active=True)[0].lower()

    def test_cpu_prefers_the_smaller_model(self):
        from sahaay.llm import order_candidates

        assert "1b" in order_candidates(self.candidates(), npu_active=False)[0].lower()

    def test_no_candidate_is_lost(self):
        from sahaay.llm import order_candidates

        c = self.candidates()
        for npu in (True, False):
            assert sorted(order_candidates(c, npu)) == sorted(c)

    def test_empty_list_is_safe(self):
        from sahaay.llm import order_candidates

        assert order_candidates([], npu_active=False) == []

    def test_unrecognised_names_keep_their_position(self):
        from sahaay.llm import order_candidates

        c = ["mystery_model", "llama_3_2_1b_instruct_genai"]
        assert order_candidates(c, npu_active=False)[0] == "llama_3_2_1b_instruct_genai"


class TestSeedGlossaryLookup:
    """Inflected forms have to reach the entry that explains them.

    A lecture says "diagonalization"; the glossary stores "diagonalize".
    Exact lookup missed it, so the sidebar offered "install the language
    model for a full explanation" for a term it could already explain - and
    listed the two forms as separate entries for the same idea.
    """

    def test_exact_term_resolves_to_itself(self):
        from sahaay.llm import SEED_GLOSSARY, seed_entry

        key, explanation = seed_entry("determinant")
        assert key == "determinant"
        assert explanation == SEED_GLOSSARY["determinant"]

    @pytest.mark.parametrize(
        ("word", "canonical"),
        [
            ("eigenvalues", "eigenvalue"),
            ("Eigenvalues", "eigenvalue"),
            ("diagonalization", "diagonalize"),
            ("diagonalizing", "diagonalize"),
            ("diagonalized", "diagonalize"),
        ],
    )
    def test_inflections_reach_the_stored_entry(self, word, canonical):
        from sahaay.llm import seed_entry

        key, explanation = seed_entry(word)
        assert key == canonical
        assert explanation, f"{word} resolved to {canonical} but got no explanation"

    def test_unknown_words_are_not_invented(self):
        """A wrong entry costs more than a missing one."""
        from sahaay.llm import seed_entry

        key, explanation = seed_entry("photosynthesis")
        assert key == "photosynthesis"
        assert explanation is None

    def test_both_forms_produce_one_sidebar_entry(self):
        from sahaay.llm import HeuristicLlm

        result = HeuristicLlm().generate(
            "<transcript>We can diagonalize the matrix. "
            "Diagonalization turns a hard repeated multiplication into a simple one."
            "</transcript>"
        )
        terms = [line.split("::")[0].strip() for line in result.text.splitlines()]
        assert terms.count("diagonalize") == 1
        assert "diagonalization" not in terms

    def test_a_known_term_never_falls_back_to_the_placeholder(self):
        from sahaay.llm import HeuristicLlm

        result = HeuristicLlm().generate(
            "<transcript>Diagonalization is the goal.</transcript>"
        )
        assert "install the language model" not in result.text
