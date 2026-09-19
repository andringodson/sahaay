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
