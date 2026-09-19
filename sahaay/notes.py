"""End-of-session artefacts: structured notes, a glossary, and a self-quiz.

A live caption disappears the moment it scrolls. What a student actually
keeps is this file. It is written as plain Markdown into ``sessions/`` so it
opens in any editor, syncs to whatever they already use, and survives the
app being uninstalled.

The quiz is not decoration. Retrieval practice is the single best-evidenced
study technique there is, and generating the questions locally means a
student with no data plan still gets it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import SUPPORTED_LANGUAGES, NotesConfig
from .glossary import GlossEntry
from .llm import LlmBackend

log = logging.getLogger(__name__)

# A bullet ending on one of these was cut off mid-sentence by the token
# budget. A bullet ending on a content word is simply short, and short
# bullets are exactly what a "Topics" list is made of.
_DANGLING_WORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with", "from", "by", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must", "that", "which", "who", "whom", "whose", "this", "these", "those", "it", "its", "as", "if", "than", "then", "when", "where", "while", "because", "so", "such", "into", "onto", "about", "over", "under", "between"]
)


SUMMARY_PROMPT = """Below is the transcript of a university lecture.

Write study notes with exactly these sections and nothing else:

## Topics
3 to 6 bullet points naming what was covered.

## Key points
4 to 8 bullet points a student should remember for an exam.

## Follow up
1 to 3 things the lecturer said would come later, or that the student \
should revise. If none were mentioned, write "None mentioned".

Keep every bullet under 20 words. Use the lecture's own terminology.

<transcript>
{transcript}
</transcript>
"""

QUIZ_PROMPT = """Below is the transcript of a university lecture.

Write exactly {n} short self-test questions covering the main ideas.

Rules:
- Write each question on its own line, then its answer on the very next line.
- Separate each question/answer pair with a blank line.
- The answer must be a direct factual sentence.
- Do NOT write multiple-choice options. Never write "A)", "B)", "C)" or "D)".
- Only state things the transcript actually says.

Example of the required shape:

What is the characteristic equation used for?
It is used to find the eigenvalues of a matrix.

<transcript>
{transcript}
</transcript>
"""


@dataclass
class CaptionRecord:
    """One finalised caption plus whatever was derived from it."""

    index: int
    text: str
    translation: str = ""
    language: str | None = None
    start_s: float = 0.0
    end_s: float = 0.0
    asr_latency_ms: float = 0.0
    rtf: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "translation": self.translation,
            "language": self.language,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "asr_latency_ms": round(self.asr_latency_ms, 1),
            "rtf": round(self.rtf, 3),
        }


@dataclass
class QuizItem:
    question: str
    answer: str


@dataclass
class SessionNotes:
    title: str
    started_at: dt.datetime
    ended_at: dt.datetime
    captions: list[CaptionRecord]
    glossary: list[GlossEntry]
    summary_markdown: str = ""
    quiz: list[QuizItem] = field(default_factory=list)
    target_language: str = "hi"
    device: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def duration_minutes(self) -> float:
        return (self.ended_at - self.started_at).total_seconds() / 60.0


class NotesWriter:
    """Builds the session artefacts and writes them to disk."""

    def __init__(self, llm: LlmBackend, cfg: NotesConfig, sessions_dir: Path):
        self.llm = llm
        self.cfg = cfg
        self.sessions_dir = Path(sessions_dir)

    # -- generation --------------------------------------------------------

    def build(
        self,
        captions: list[CaptionRecord],
        glossary: list[GlossEntry],
        target_language: str,
        started_at: dt.datetime,
        device: dict | None = None,
        metrics: dict | None = None,
        title: str | None = None,
    ) -> SessionNotes:
        ended_at = dt.datetime.now()
        transcript = " ".join(c.text for c in captions).strip()

        notes = SessionNotes(
            title=title or self._infer_title(captions, started_at),
            started_at=started_at,
            ended_at=ended_at,
            captions=captions,
            glossary=glossary,
            target_language=target_language,
            device=device or {},
            metrics=metrics or {},
        )

        if not transcript:
            notes.summary_markdown = "_No speech was captured in this session._"
            return notes

        # Long lectures blow past the model's context. Take the head and tail
        # rather than truncating: the opening states what the lecture is
        # about and the close states what to revise, and both matter more
        # than the middle for a summary.
        excerpt = self._excerpt(transcript)

        # The heuristic backend extracts terms; it cannot summarise or write
        # questions. Feeding it the summary prompt returns glossary lines,
        # which would render as a "summary" that just repeats the glossary
        # section below it. Go straight to the structured fallback instead.
        can_summarise = self.cfg.enabled and getattr(self.llm, "name", "") != "heuristic"

        if can_summarise:
            try:
                notes.summary_markdown = self._clean(
                    self.llm.generate(SUMMARY_PROMPT.format(transcript=excerpt), max_new_tokens=420).text
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("summary generation failed: %s", exc)

            try:
                quiz_raw = self.llm.generate(
                    QUIZ_PROMPT.format(n=self.cfg.quiz_questions, transcript=excerpt),
                    max_new_tokens=400,
                ).text
                notes.quiz = self._parse_quiz(quiz_raw)
            except Exception as exc:  # noqa: BLE001
                log.warning("quiz generation failed: %s", exc)

        if not notes.summary_markdown:
            notes.summary_markdown = self._fallback_summary(captions, glossary)
        return notes

    @staticmethod
    def _excerpt(transcript: str, budget_chars: int = 6000) -> str:
        if len(transcript) <= budget_chars:
            return transcript
        head = transcript[: budget_chars // 2]
        tail = transcript[-budget_chars // 2 :]
        return f"{head}\n\n[...middle of lecture omitted...]\n\n{tail}"

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise model Markdown into the subset the UI renders.

        Two things bite here. Models write "•" or "–" for bullets, which the
        renderer does not treat as list items, so the notes came out as one
        run-on paragraph. And when generation hits the token budget the last
        bullet arrives half-written ("Real eigenvalues are "), which reads as
        a bug rather than a limit.
        """
        # Models often restate the prompt or open with "Sure!".
        text = re.sub(r"^\s*(sure|here (is|are)|certainly)[^\n]*\n", "", text, flags=re.I)

        lines = [re.sub(r"^\s*[•‣▪–—]\s+", "- ", ln) for ln in text.strip().splitlines()]

        # Drop a trailing bullet only when it was genuinely cut mid-sentence.
        # Length is the wrong signal - "- Eigenvalues" is a perfectly good
        # entry in a Topics list, and an earlier length-based rule deleted
        # every short bullet. Ending on a function word is the real tell:
        # "- Real eigenvalues are" was cut, "- Eigenvalues" was not.
        if lines:
            last = lines[-1].strip()
            if last.startswith("- ") and not last.endswith((".", "!", "?", ":", ")", "।")):
                words = last[2:].split()
                if words and words[-1].lower() in _DANGLING_WORDS:
                    lines.pop()

        return "\n".join(lines).strip()

    # Leading decoration on a question line: "Q ::", "Q:", "1.".
    # Bullets are deliberately absent - an answer beginning "- " would
    # otherwise be mistaken for a new question.
    _Q_MARKER = re.compile(r"^\s*(?:Q\s*::|Q\s*[:.)]|\d+\s*[.)])\s*", re.I)
    # Models slip into multiple choice whatever the prompt says.
    _MULTIPLE_CHOICE = re.compile(r"\b[B-D]\)\s")
    # Leading decoration on an answer line: "A:", "Answer -", "A ::".
    _A_MARKER = re.compile(r"^\s*(?:A\s*::|A\s*[:.)]|answer\s*[:.\-]?)\s*", re.I)

    @classmethod
    def _parse_quiz(cls, text: str) -> list[QuizItem]:
        """Parse a quiz out of whatever shape the model produced.

        The prompt asks for ``Q :: question :: answer``. Llama 3.2 gave three
        different shapes across three runs: that one, ``Q :: question`` with
        the answer on the next line, and - once a repetition penalty was
        added - no markers at all, just a question line followed by an answer
        line. A parser keyed to one shape returned zero items from output
        that was otherwise perfectly good.

        So parse structurally instead: split on blank lines, treat the first
        line of each block as the question and the rest as the answer, and
        strip whatever decoration is present. That survives a model swap,
        which prompt-wording tricks do not.
        """
        items: list[QuizItem] = []
        pending: str | None = None
        answer_lines: list[str] = []

        def flush() -> None:
            nonlocal pending, answer_lines
            if pending:
                answer = cls._A_MARKER.sub("", " ".join(answer_lines)).strip()
                # An "answer" that is really a list of options is worse than
                # no question at all - the card would show no answer.
                if answer and not cls._MULTIPLE_CHOICE.search(answer):
                    items.append(QuizItem(question=pending, answer=answer))
            pending, answer_lines = None, []

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                # Blank lines carry no meaning: models put one between the
                # question and its answer as readily as between pairs.
                continue

            head = cls._Q_MARKER.sub("", line).strip()
            marked = head != line

            question, sep, inline = head.partition("::")
            if sep and question.strip() and inline.strip():
                flush()
                pending, answer_lines = question.strip(), [inline.strip()]
                flush()
                continue

            if head.endswith("?") or (marked and not answer_lines):
                flush()
                pending = head
            elif pending:
                answer_lines.append(head)

        flush()
        return items

    def _fallback_summary(self, captions: list[CaptionRecord], glossary: list[GlossEntry]) -> str:
        """No-LLM summary: the glossary terms plus the longest caption lines.

        Crude, but a student who ran this with no weights downloaded still
        gets something back rather than an empty file.
        """
        lines = ["## Topics", ""]
        if glossary:
            lines += [f"- {g.term}" for g in glossary[:6]]
        else:
            lines.append("- (no technical terms detected)")
        lines += ["", "## Key points", ""]
        longest = sorted(captions, key=lambda c: len(c.text), reverse=True)[:6]
        for c in sorted(longest, key=lambda c: c.index):
            lines.append(f"- {c.text}")
        lines += ["", "## Follow up", "", "None mentioned.", ""]
        lines.append("_Generated without an LLM - install the model for better notes._")
        return "\n".join(lines)

    @staticmethod
    def _infer_title(captions: list[CaptionRecord], started_at: dt.datetime) -> str:
        stamp = started_at.strftime("%Y-%m-%d %H:%M")
        if not captions:
            return f"Session {stamp}"
        first = captions[0].text.strip()
        words = first.split()[:7]
        return f"{' '.join(words)} - {stamp}" if words else f"Session {stamp}"

    # -- output ------------------------------------------------------------

    def render_markdown(self, notes: SessionNotes) -> str:
        lang_name = SUPPORTED_LANGUAGES.get(notes.target_language, {}).get(
            "name", notes.target_language
        )
        out: list[str] = [
            f"# {notes.title}",
            "",
            f"**Date** {notes.started_at:%d %b %Y, %H:%M} · "
            f"**Duration** {notes.duration_minutes:.0f} min · "
            f"**Captions** {len(notes.captions)}",
            "",
        ]
        if notes.device:
            out += [
                f"_Processed entirely on-device: {notes.device.get('provider_label', 'unknown')}_",
                "",
            ]

        out += [notes.summary_markdown, ""]

        if notes.glossary:
            out += ["## Glossary", "", f"_Explanations in {lang_name}._", ""]
            for g in notes.glossary:
                out.append(f"- **{g.term}** — {g.explanation}")
            out.append("")

        if notes.quiz:
            out += ["## Self-test", ""]
            for i, q in enumerate(notes.quiz, 1):
                out += [f"{i}. {q.question}", f"   <details><summary>Answer</summary>{q.answer}</details>", ""]

        out += ["## Transcript", ""]
        for c in notes.captions:
            ts = f"{int(c.start_s) // 60:02d}:{int(c.start_s) % 60:02d}"
            out.append(f"**[{ts}]** {c.text}")
            if c.translation and c.translation != c.text:
                out.append(f"> {c.translation}")
            out.append("")

        if notes.metrics:
            out += ["## Performance", "", "```json", json.dumps(notes.metrics, indent=2), "```", ""]

        return "\n".join(out)

    def save(self, notes: SessionNotes) -> Path:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", notes.title).strip("-").lower()[:60]
        stamp = notes.started_at.strftime("%Y%m%d-%H%M")
        base = self.sessions_dir / f"{stamp}-{slug or 'session'}"

        md_path = base.with_suffix(".md")
        md_path.write_text(self.render_markdown(notes), encoding="utf-8")

        # The JSON sidecar is what the benchmark harness and any future
        # export path read; Markdown is for the human.
        json_path = base.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "title": notes.title,
                    "started_at": notes.started_at.isoformat(),
                    "ended_at": notes.ended_at.isoformat(),
                    "target_language": notes.target_language,
                    "device": notes.device,
                    "metrics": notes.metrics,
                    "captions": [c.to_dict() for c in notes.captions],
                    "glossary": [g.to_dict() for g in notes.glossary],
                    "quiz": [{"q": q.question, "a": q.answer} for q in notes.quiz],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("session saved: %s", md_path)
        return md_path
