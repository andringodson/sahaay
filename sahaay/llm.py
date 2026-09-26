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

from .config import resolve_model_id

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

    def __init__(self, model_dir: Path, cpu_threads: int = 0):
        import onnxruntime_genai as og  # type: ignore

        # Recorded so the benchmark can name what it measured. Without it the
        # table said "-" for the LLM row, the same gap that made the ASR
        # number unverifiable.
        self.model_dir = Path(model_dir)
        self._og = og
        t0 = time.perf_counter()
        self._model = self._load(og, model_dir, cpu_threads)
        self._tokenizer = og.Tokenizer(self._model)
        self._chat_template = hasattr(self._tokenizer, "apply_chat_template")
        self.available = True
        self.cpu_threads = cpu_threads
        # One generation at a time: the NPU has a single HTP context and
        # concurrent generate() calls would serialise unpredictably anyway.
        self._lock = threading.Lock()
        log.info("LLM loaded from %s in %.1fs", model_dir.name, time.perf_counter() - t0)

    @staticmethod
    def _load(og, model_dir: Path, cpu_threads: int):  # noqa: ANN001, ANN205
        """Load the model, throttled when it shares a CPU with the captions.

        GlossaryConfig has always said the glossary runs "at low priority"
        and that on CPU "we throttle hard so captions never stall". Nothing
        did: the model claimed every core, and so did Whisper and NLLB,
        running at the same moment. Capping its threads is that throttle.

        On the NPU none of this applies - cpu_threads is 0 and the model loads
        exactly as it did.
        """
        if not cpu_threads:
            return og.Model(str(model_dir))
        import json

        try:
            # Thread counts only. onnxruntime-genai 0.11 rejects
            # "config_entries" in session_options outright ("Unknown value"),
            # so the spin-wait switch the caption models get cannot be passed
            # here - and the whole overlay failing took the thread cap down
            # with it, silently, until a benchmark log showed the fallback.
            config = og.Config(str(model_dir))
            config.overlay(json.dumps({"model": {"decoder": {"session_options": {
                "intra_op_num_threads": cpu_threads,
                "inter_op_num_threads": 1,
            }}}}))
            return og.Model(config)
        except Exception as exc:  # noqa: BLE001
            # An older runtime without overlay() or config_entries still gets
            # a working glossary, just an unthrottled one.
            log.warning("could not throttle the glossary model (%s); loading it plainly", exc)
            return og.Model(str(model_dir))

    # Llama 3 chat format, used when the runtime cannot apply the model's own
    # template. Instruction-tuned models only follow instructions inside this
    # framing; a bare prompt makes them continue the text instead. Observed:
    # "Reply with exactly the word: OK" returned ", and then the conversation
    # continues.\n\n**Conversation:**" - fluent, and completely useless.
    _LLAMA3_TEMPLATE = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    def _as_chat(self, prompt: str) -> str:
        if self._chat_template:
            try:
                import json

                messages = json.dumps([{"role": "user", "content": prompt}])
                return self._tokenizer.apply_chat_template(
                    messages=messages, add_generation_prompt=True
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("apply_chat_template failed (%s); using Llama 3 format", exc)
        return self._LLAMA3_TEMPLATE.format(prompt=prompt)

    def generate(self, prompt: str, max_new_tokens: int = 160) -> LlmResult:
        og = self._og
        with self._lock:
            t0 = time.perf_counter()
            input_ids = self._tokenizer.encode(self._as_chat(prompt))

            params = og.GeneratorParams(self._model)
            # Greedy, because we want the same term to get the same
            # explanation every time it is spoken - temperature would make
            # the sidebar contradict itself across a lecture.
            #
            # But greedy decoding on a small model loops. Observed with
            # Llama 3.2 1B: the notes repeated four bullets for 30 lines
            # until the token budget ran out. repetition_penalty and
            # no_repeat_ngram_size are what make greedy usable here.
            options = {
                "max_length": len(input_ids) + max_new_tokens,
                "do_sample": False,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 12,
            }
            try:
                params.set_search_options(**options)
            except Exception:  # noqa: BLE001 - older GenAI rejects unknown keys
                try:
                    params.set_search_options(
                        max_length=options["max_length"], do_sample=False
                    )
                except Exception:  # noqa: BLE001
                    params.set_search_options(max_length=options["max_length"])

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


def seed_entry(term: str) -> tuple[str, str | None]:
    """Resolve a word to its seeded glossary entry, allowing for inflection.

    A lecture says "diagonalization"; the glossary stores "diagonalize".
    Exact lookup missed it and the sidebar showed "install the language
    model for a full explanation" next to a term the glossary could in fact
    explain - and listed "diagonalize" separately, so the same idea appeared
    twice.

    Returns the canonical key and its explanation, or the word itself and
    None when nothing matches. Same approach as
    :meth:`~sahaay.translate.TermProtector._is_protected`: strip the handful
    of endings that actually occur rather than take on a stemmer.
    """
    word = term.lower().strip("-")
    if word in SEED_GLOSSARY:
        return word, SEED_GLOSSARY[word]

    candidates: list[str] = []
    for suffix in ("s", "es", "ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            # English drops the silent e before -ing/-ed; put it back.
            candidates += [stem, stem + "e"]
    for suffix, replacement in (("ization", "ize"), ("isation", "ise"), ("ation", "ate")):
        if word.endswith(suffix):
            candidates.append(word[: -len(suffix)] + replacement)
    if word.endswith("ices"):  # matrices -> matrix
        candidates.append(word[:-4] + "ix")

    for candidate in candidates:
        if candidate in SEED_GLOSSARY:
            return candidate, SEED_GLOSSARY[candidate]

    return word, None


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

        found: dict[str, str] = {}
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text):
            key = word.lower().strip("-")
            if key in _STOPWORDS:
                continue
            # Inflections collapse onto the canonical term, so "diagonalize"
            # and "diagonalization" are one sidebar entry, not two.
            canonical, explanation = seed_entry(key)
            if canonical in found:
                continue
            # Known terms first. Unknown ones only clear the bar if the
            # morphology is strongly scientific - a wrong entry costs more
            # credibility than a missing one costs coverage.
            if explanation is not None:
                found[canonical] = explanation
            elif self._looks_technical(key):
                found[canonical] = (
                    "Technical term from this lecture - install the language "
                    "model for a full explanation."
                )
            if len(found) >= 3:
                break

        lines = [f"{term} :: {explanation}" for term, explanation in found.items()]
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


def find_genai_model(root: Path) -> Path | None:
    """Locate the directory ONNX Runtime GenAI should be pointed at.

    GenAI wants the folder containing ``genai_config.json``, not the repo
    root. Published builds nest it by target, e.g.
    ``cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/``, so pointing at the
    downloaded directory directly fails. Prefer a CPU/NPU build over CUDA,
    which would never load on a Snapdragon laptop anyway.
    """
    if not root.exists():
        return None
    if (root / "genai_config.json").exists():
        return root

    candidates = sorted(p.parent for p in root.rglob("genai_config.json"))
    if not candidates:
        return None
    for path in candidates:
        if "cuda" not in str(path).lower():
            return path
    return candidates[0]


def order_candidates(candidates: list[str], npu_active: bool) -> list[str]:
    """Put the model that suits this device first.

    Measured on an x86 CPU: the 3B runs at 7.4 tok/s against the 1B's 16.9,
    which is the expected ratio - but the 3B needs ~3.5 GB resident, and on a
    machine short of RAM it degrades far past that. One 146-token glossary
    call was observed taking 35 minutes under memory pressure, versus about
    six seconds for the same call on the 1B.

    So on a CPU-only machine the 1B is the right default: a glossary entry
    that arrives after the lecture has ended is not a glossary entry. With
    the NPU active the larger model is preferred, because that is the
    hardware it was chosen for and its explanations are visibly better.
    """
    if npu_active:
        return list(candidates)

    def cpu_rank(name: str) -> tuple[int, int]:
        lowered = name.lower()
        # Smaller first on CPU; anything unrecognised keeps its position.
        size = 0 if "1b" in lowered else 1 if "3b" in lowered else 2
        return (size, candidates.index(name))

    return sorted(candidates, key=cpu_rank)


def create_llm(
    models_dir: Path,
    model_id: str,
    mock: bool = False,
    candidates: list[str] | None = None,
    npu_active: bool = False,
    cpu_threads: int = 0,
) -> LlmBackend:
    """Walk the degradation ladder and return the best backend available.

    Resolution happens here rather than in each caller. It used to be the
    caller's job, and the benchmark harness duly passed the literal string
    "auto" straight through, so the LLM stage silently reported "weights not
    downloaded" while the model sat on disk.
    """
    if mock:
        log.info("LLM: heuristic backend (mock mode)")
        return HeuristicLlm()

    if model_id == "auto" or not model_id:
        ordered = order_candidates(candidates or [], npu_active)
        model_id = resolve_model_id(models_dir, model_id, ordered)
        log.info("glossary model: %s (%s)", model_id, "NPU" if npu_active else "CPU")

    model_dir = find_genai_model(models_dir / model_id)
    if model_dir is not None:
        try:
            # Throttle only when it shares the CPU with the captions.
            return GenAiLlm(model_dir, cpu_threads=0 if npu_active else cpu_threads)
        except ImportError:
            log.warning(
                "onnxruntime-genai not installed; glossary falls back to heuristics.\n"
                "  pip install onnxruntime-genai"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM failed to load (%s); falling back to heuristics", exc)
    else:
        # Log the directory we searched, not model_dir - it is None here.
        log.info(
            "no genai_config.json under %s; using heuristic glossary",
            models_dir / model_id,
        )
    return HeuristicLlm()
