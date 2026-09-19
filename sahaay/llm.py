"""Local LLM backend shared by the glossary sidebar and the notes writer.

On a Snapdragon HP PC this is Llama 3.2 3B executing on the Hexagon NPU via
ONNX Runtime GenAI, using the prebuilt ``hexagon-npu-assets`` package so no
QNN context binary has to be compiled locally. Everywhere else it degrades:
GenAI on CPU/DirectML, then a rule-based extractor that needs no model at
all.

The degradation ladder is the point. A judge running this on an x86 laptop
still sees a working glossary; the NPU makes it fast enough to run *while*
Whisper is transcribing, which is the concurrency story the whole product
rests on.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class LlmResult:
    text: str
    latency_ms: float
    tokens: int
    backend: str

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / (self.latency_ms / 1000.0) if self.latency_ms > 0 else 0.0


class LlmBackend:
    name = "base"
    available = False

    def generate(self, prompt: str, max_new_tokens: int = 160) -> LlmResult:  # pragma: no cover
        raise NotImplementedError


class GenAiLlm(LlmBackend):
    """ONNX Runtime GenAI, which handles KV cache and sampling for us.

    Writing our own decode loop for a 3B model would mean managing 28 layers
    of KV cache by hand across a provider boundary. GenAI already does this
    and supports the Hexagon NPU asset format directly.
    """

    name = "onnxruntime-genai"

    def __init__(self, model_dir: Path):
        import onnxruntime_genai as og  # type: ignore

        self._og = og
        t0 = time.perf_counter()
        self._model = og.Model(str(model_dir))
        self._tokenizer = og.Tokenizer(self._model)
        self.available = True
        # One generation at a time: the NPU has a single HTP context and
        # concurrent generate() calls would serialise unpredictably anyway.
        self._lock = threading.Lock()
        log.info("LLM loaded from %s in %.1fs", model_dir.name, time.perf_counter() - t0)

    def generate(self, prompt: str, max_new_tokens: int = 160) -> LlmResult:
        og = self._og
        with self._lock:
            t0 = time.perf_counter()
            input_ids = self._tokenizer.encode(prompt)

            params = og.GeneratorParams(self._model)
            # Greedy: we want stable, repeatable glossary definitions, not
            # creative ones. Temperature here would produce a different
            # explanation for the same term each time it appears.
            try:
                params.set_search_options(
                    max_length=len(input_ids) + max_new_tokens,
                    do_sample=False,
                )
            except Exception:  # noqa: BLE001 - older GenAI signature
                params.set_search_options(max_length=len(input_ids) + max_new_tokens)

            generator = og.Generator(self._model, params)
            try:
                generator.append_tokens(input_ids)
            except AttributeError:  # older API
                params.input_ids = input_ids
                generator = og.Generator(self._model, params)

            out: list[int] = []
            while not generator.is_done() and len(out) < max_new_tokens:
                generator.generate_next_token()
                seq = generator.get_sequence(0)
                if len(seq) > len(input_ids):
                    out = list(seq[len(input_ids):])

            text = self._tokenizer.decode(out) if out else ""
            latency_ms = (time.perf_counter() - t0) * 1000.0

        return LlmResult(text=text.strip(), latency_ms=latency_ms, tokens=len(out), backend=self.name)


# Terms that look technical but are just discourse, plus the closed-class
# words that survive a capitalisation filter. Without this the sidebar
# fills with "The", "Now", "Today".
_STOPWORDS = {
    "the", "this", "that", "these", "those", "and", "but", "for", "with",
    "from", "into", "onto", "about", "today", "now", "next", "week", "class",
    "lecture", "chapter", "question", "answer", "example", "problem", "let",
    "remember", "okay", "right", "so", "we", "will", "can", "you", "your",
    "our", "here", "there", "when", "then", "also", "very", "just", "like",
    # Academic register that passes the -tion/-ity morphology test but that
    # no student needs defined. Without these the sidebar fills with
    # "solution", "equation", "function" and reads as broken.
    "solution", "equation", "function", "section", "direction", "condition",
    "position", "operation", "information", "situation", "definition",
    "explanation", "application", "calculation", "relation", "notation",
    "attention", "discussion", "conclusion", "expression", "assumption",
    "important", "different", "following", "because", "therefore",
    "something", "everything", "understand", "basically", "actually",
}

# Seeded with the STEM vocabulary an Indian engineering syllabus actually
# uses. The LLM handles anything not in here; this table exists so the
# no-model fallback is still useful rather than empty.
SEED_GLOSSARY: dict[str, str] = {
    "eigenvalue": "A scalar showing how much a matrix stretches a special direction (its eigenvector).",
    "eigenvector": "A direction that a matrix only stretches or shrinks, without rotating it.",
    "determinant": "A single number from a square matrix; zero means the matrix squashes space flat.",
    "diagonalize": "Rewriting a matrix in a basis where it becomes a simple diagonal scaling.",
    "singular": "A matrix with determinant zero, so it cannot be inverted.",
    "svd": "Singular Value Decomposition: splits any matrix into rotate-scale-rotate.",
    "rank": "The number of genuinely independent directions a matrix maps onto.",
    "nullspace": "All the vectors a matrix crushes to zero.",
    "gradient": "The direction of steepest increase of a function.",
    "convolution": "Sliding a small filter across data to detect a local pattern.",
    "entropy": "How uncertain or disordered a distribution is.",
    "backpropagation": "The rule for pushing error backwards through a network to update weights.",
    "overfitting": "When a model memorises training data and fails on new data.",
    "quantization": "Storing numbers with fewer bits so a model runs faster and smaller.",
    "mitosis": "Cell division producing two identical daughter cells.",
    "meiosis": "Cell division producing four cells with half the chromosomes, for reproduction.",
    "enthalpy": "Total heat content of a system at constant pressure.",
    "catalyst": "A substance that speeds up a reaction without being consumed.",
    "impedance": "Opposition to alternating current, combining resistance and reactance.",
    "capacitance": "How much charge a component stores per volt applied.",
    "throughput": "How much work a system completes per unit time.",
    "latency": "The delay between asking for something and getting it.",
}


class HeuristicLlm(LlmBackend):
    """Model-free fallback.

    Not a toy: it uses a seeded STEM glossary plus a morphology filter, so
    the sidebar still populates on a machine with zero model weights
    downloaded. It cannot explain a term it has never seen - that is exactly
    what the 3B model buys you, and the UI labels which one produced a line.
    """

    name = "heuristic"

    def __init__(self) -> None:
        self.available = True

    def generate(self, prompt: str, max_new_tokens: int = 160) -> LlmResult:
        t0 = time.perf_counter()
        # The prompt embeds the transcript between markers; pull it back out.
        m = re.search(r"<transcript>(.*?)</transcript>", prompt, re.S)
        text = m.group(1) if m else prompt

        found: list[str] = []
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text):
            key = word.lower().strip("-")
            if key in _STOPWORDS or key in found:
                continue
            # Known terms first. Unknown ones only clear the bar if the
            # morphology is strongly scientific - a wrong entry costs more
            # credibility than a missing one costs coverage.
            if key in SEED_GLOSSARY or self._looks_technical(key):
                found.append(key)
            if len(found) >= 3:
                break

        lines = [
            f"{term} :: {SEED_GLOSSARY.get(term, 'Technical term from this lecture - install the language model for a full explanation.')}"
            for term in found
        ]
        return LlmResult(
            text="\n".join(lines),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens=len(lines),
            backend=self.name,
        )

    @staticmethod
    def _looks_technical(word: str) -> bool:
        """Latin/Greek scientific morphology.

        Only the suffixes that are almost exclusively scientific. ``-tion``
        and ``-ity`` are deliberately excluded: they match far more ordinary
        prose than jargon, and precision matters more than recall here.
        """
        suffixes = (
            "osis", "itis", "ology", "ometry", "onomy", "genesis", "lysis",
            "phobia", "otype", "plasm", "hedron", "morphism", "ization",
        )
        return len(word) >= 9 and word.endswith(suffixes)


def create_llm(models_dir: Path, model_id: str, mock: bool = False) -> LlmBackend:
    """Walk the degradation ladder and return the best backend available."""
    if mock:
        log.info("LLM: heuristic backend (mock mode)")
        return HeuristicLlm()

    model_dir = models_dir / model_id
    if model_dir.exists():
        try:
            return GenAiLlm(model_dir)
        except ImportError:
            log.warning(
                "onnxruntime-genai not installed; glossary falls back to heuristics.\n"
                "  pip install onnxruntime-genai"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM failed to load (%s); falling back to heuristics", exc)
    else:
        log.info("LLM weights not present at %s; using heuristic glossary", model_dir)
    return HeuristicLlm()
