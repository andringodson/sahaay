"""Word error rate against speech with known ground truth.

The README said accuracy was not measured. This measures it, within a limit
that is stated up front rather than buried:

**The audio is synthesised by Windows SAPI, not spoken by people.** That
makes ground truth exact and the run reproducible on any Windows machine
with no dataset to download, and it is a fair test of the decoding path -
mel front end, KV cache, language handling, detokenisation. It is *not* a
substitute for a real corpus: TTS has no accents, no crosstalk, no room
reverb and no disfluency, so the numbers here are a floor, not a forecast.

For the code-mixed lines the limitation is sharper still. Only an en-US
voice is installed, so romanised Hindi is read with English phonetics. That
tests whether the pipeline preserves technical terms through non-English
word sequences; it does not tell you how the model handles a Hindi speaker.
Both facts appear in the generated report.

    python scripts/accuracy.py --write
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahaay.config import load_config  # noqa: E402

AUDIO_DIR = REPO_ROOT / "testaudio"

# Numbers get spoken and transcribed inconsistently ("zero" vs "0"), which is
# a formatting difference, not a recognition error.
NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}


@dataclass
class Utterance:
    name: str
    text: str
    kind: str  # "english" | "code-mixed"
    # Terms that must survive verbatim. Losing one is the failure the whole
    # translation stage exists to prevent, so it is scored separately.
    terms: list[str] = field(default_factory=list)


CORPUS: list[Utterance] = [
    Utterance(
        "en1", "So today we will start with eigenvalues and eigenvectors.",
        "english", ["eigenvalues", "eigenvectors"],
    ),
    Utterance(
        "en2", "The determinant of matrix A must be zero for a non trivial solution.",
        "english", ["determinant", "matrix"],
    ),
    Utterance(
        "en3", "This is called the characteristic equation of the matrix.",
        "english", ["characteristic", "equation"],
    ),
    Utterance(
        "en4", "Symmetric matrices always have real eigenvalues.",
        "english", ["symmetric", "eigenvalues"],
    ),
    Utterance(
        "cm1", "Aaj hum eigenvalues aur eigenvectors padhenge.",
        "code-mixed", ["eigenvalues", "eigenvectors"],
    ),
    Utterance(
        "cm2", "Matrix A ka determinant zero hoga tabhi non trivial solution milega.",
        "code-mixed", ["determinant", "solution"],
    ),
    Utterance(
        "cm3", "Yeh characteristic equation kehlata hai.",
        "code-mixed", ["characteristic", "equation"],
    ),
]


def normalise(text: str) -> list[str]:
    """Lower-case, strip punctuation, spell out small numbers."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = []
    for word in text.split():
        words.append(NUMBER_WORDS.get(word, word))
    return words


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over words - the numerator of WER."""
    if not ref:
        return len(hyp)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(
                previous[j - 1] if r == h
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    ref, hyp = normalise(reference), normalise(hypothesis)
    errors = edit_distance(ref, hyp)
    return (errors / len(ref) if ref else 0.0), errors, len(ref)


def synthesise(utterances: list[Utterance]) -> dict[str, Path]:
    """Render each line to a 16 kHz mono WAV with Windows SAPI."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}

    for utt in utterances:
        path = AUDIO_DIR / f"wer_{utt.name}.wav"
        if path.exists():
            made[utt.name] = path
            continue

        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
            "16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
            "[System.Speech.AudioFormat.AudioChannel]::Mono); "
            f"$s.SetOutputToWaveFile('{path}', $f); $s.Rate = -2; "
            f"$s.Speak('{utt.text}'); $s.SetOutputToNull(); $s.Dispose()"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True, capture_output=True, timeout=120,
            )
            made[utt.name] = path
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not synthesise {utt.name}: {exc}")
    return made


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def render_markdown(rows: list[dict], totals: dict, device: dict, model: str) -> str:
    now = dt.datetime.now().strftime("%d %b %Y")
    lines = [
        "# Transcription accuracy",
        "",
        f"Measured {now} by `scripts/accuracy.py` on **{device.get('provider_label','?')}**, "
        f"model `{model}`.",
        "",
        "## Results",
        "",
        "| Set | Utterances | WER | Technical terms kept |",
        "|---|---:|---:|---:|",
    ]
    for kind in ("english", "code-mixed"):
        t = totals.get(kind)
        if not t:
            continue
        lines.append(
            f"| {kind} | {t['count']} | **{t['wer']:.1%}** | "
            f"{t['terms_kept']}/{t['terms_total']} |"
        )
    overall = totals.get("overall")
    if overall:
        lines.append(
            f"| **all** | {overall['count']} | **{overall['wer']:.1%}** | "
            f"{overall['terms_kept']}/{overall['terms_total']} |"
        )

    lines += ["", "## Per utterance", "", "| Ref | Hypothesis | WER |", "|---|---|---:|"]
    for r in rows:
        lines.append(
            f"| {r['reference']} | {r['hypothesis'] or '_(nothing)_'} | {r['wer']:.1%} |"
        )

    lines += [
        "",
        "## What this measures, and what it does not",
        "",
        "**The audio is synthesised by Windows SAPI, not spoken by people.** That",
        "makes ground truth exact and lets anyone reproduce this on a Windows",
        "machine with no dataset to download. It is a fair test of the decoding",
        "path - mel front end, KV cache, language selection, detokenisation.",
        "",
        "It is **not** a substitute for a real corpus. Synthetic speech has no",
        "accent, no room reverb, no crosstalk and no disfluency, so these numbers",
        "are a floor rather than a forecast. Expect materially worse on a real",
        "lecture recording.",
        "",
        "For the code-mixed set the caveat is stronger: only an **en-US** voice is",
        "installed, so romanised Hindi is read with English phonetics. That tests",
        "whether the pipeline carries technical terms through non-English word",
        "sequences. It tells you nothing about how the model handles an actual",
        "Hindi speaker, and the WER for that set should be read as a lower bound on",
        "difficulty, not an estimate of field accuracy.",
        "",
        "**Technical terms kept** is scored separately because it is the metric the",
        "product actually depends on. A student can work around a garbled function",
        "word; a garbled `eigenvalue` is the one that breaks the link to the",
        "textbook.",
        "",
        "Numbers are normalised before scoring (\"0\" and \"zero\" count as the same",
        "word), since that is a formatting difference rather than a recognition",
        "error.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure word error rate.")
    ap.add_argument("--write", action="store_true", help="write docs/ACCURACY.md")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--language", help="force a language instead of detecting")
    args = ap.parse_args(argv)

    from sahaay.asr import create_asr
    from sahaay.config import resolve_model_id
    from sahaay.runtime import SessionFactory

    cfg = load_config()
    factory = SessionFactory(cfg.runtime)
    device = factory.report()
    model = resolve_model_id(cfg.models_dir, cfg.asr.model_id, cfg.asr.candidates)
    print(f"\n  device: {device.provider_label}   model: {model}\n")

    try:
        asr = create_asr(cfg.models_dir, factory, cfg.asr)
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return 1

    print("  synthesising reference audio...")
    audio_paths = synthesise(CORPUS)
    if not audio_paths:
        print("  no audio could be synthesised (Windows SAPI required)")
        return 1

    rows: list[dict] = []
    for utt in CORPUS:
        path = audio_paths.get(utt.name)
        if path is None:
            continue
        result = asr.transcribe(read_wav(path), language=args.language)
        rate, errors, ref_len = wer(utt.text, result.text)

        hyp_words = set(normalise(result.text))
        kept = [t for t in utt.terms if t.lower() in hyp_words]

        rows.append({
            "name": utt.name,
            "kind": utt.kind,
            "reference": utt.text,
            "hypothesis": result.text,
            "wer": rate,
            "errors": errors,
            "ref_words": ref_len,
            "terms_total": len(utt.terms),
            "terms_kept": len(kept),
            "terms_missed": [t for t in utt.terms if t not in kept],
            "language": result.language,
        })
        print(f"  {utt.name}  WER {rate:5.1%}  terms {len(kept)}/{len(utt.terms)}")
        print(f"        -> {result.text[:100]}")

    def totals_for(kind: str | None) -> dict:
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        if not sel:
            return {}
        errors = sum(r["errors"] for r in sel)
        words = sum(r["ref_words"] for r in sel)
        return {
            "count": len(sel),
            "wer": errors / words if words else 0.0,
            "terms_kept": sum(r["terms_kept"] for r in sel),
            "terms_total": sum(r["terms_total"] for r in sel),
        }

    totals = {
        "english": totals_for("english"),
        "code-mixed": totals_for("code-mixed"),
        "overall": totals_for(None),
    }

    print(f"\n  overall WER {totals['overall']['wer']:.1%}   "
          f"terms kept {totals['overall']['terms_kept']}/{totals['overall']['terms_total']}")

    md = render_markdown(rows, totals, device.to_dict(), model)
    if args.write:
        out = REPO_ROOT / "docs" / "ACCURACY.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"  wrote {out}")
    else:
        print("\n" + md)

    if args.json:
        args.json.write_text(
            json.dumps({"device": device.to_dict(), "model": model,
                        "totals": totals, "rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
