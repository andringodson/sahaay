"""Synthesise a longer lecture WAV for demos and end-to-end runs.

``testaudio/lecture.wav`` is 14 seconds, which is enough to prove the pipeline
works and not nearly enough to show it working. A demo needs a lecture with
enough jargon for the glossary to have something to do and enough length for
the RTF readout to settle.

Sentences are rendered one at a time and joined with a gap of silence, rather
than read as one block. That is deliberate: the segmenter splits on pauses,
so synthesising the pauses explicitly gives clean caption boundaries instead
of whatever SAPI happens to do between clauses.

Windows only - it uses System.Speech, the same route scripts/accuracy.py
takes, so the audio is reproducible on any Windows machine with no dataset to
download and no voice to install beyond the one that ships.

    python scripts/make_lecture_audio.py
    python scripts/make_lecture_audio.py --out testaudio/lecture_long.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 16000

# A first lecture on eigenvalues. Chosen because it is dense with terms that
# have no everyday translation - which is the case the product exists for -
# and because the same material already backs the accuracy corpus.
LECTURE = [
    "Good morning everyone. Today we will start with eigenvalues and eigenvectors.",
    "These two ideas sit at the centre of linear algebra, so it is worth going slowly.",
    "A matrix is a linear transformation. It stretches, rotates and shears space.",
    "For most vectors, applying the matrix changes both direction and length.",
    "But a few special vectors only change length. Their direction is preserved.",
    "Those vectors are the eigenvectors, and the factor they stretch by is the eigenvalue.",
    "To find them we solve the characteristic equation of the matrix.",
    "The determinant of A minus lambda times the identity must equal zero.",
    "The determinant is a single number that tells you whether the matrix squashes space flat.",
    "If it is zero, there is a non trivial solution, and that is exactly what we want.",
    "Symmetric matrices always have real eigenvalues. That is a useful guarantee.",
    "Once we have a full set of eigenvectors, we can diagonalize the matrix.",
    "Diagonalization turns a hard repeated multiplication into a simple one.",
    "We will use this next week when we look at principal component analysis.",
]

GAP_S = 0.75


def speak(text: str, path: Path, rate: int = -2) -> bool:
    """Render one sentence with SAPI. Returns False if it could not."""
    safe = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        f"{SAMPLE_RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Rate = {rate}; "
        f"$s.Speak('{safe}'); $s.SetOutputToNull(); $s.Dispose()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=True, capture_output=True, timeout=120,
        )
        return path.exists()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {exc}")
        return False


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "testaudio" / "lecture_long.wav")
    ap.add_argument("--rate", type=int, default=-2, help="SAPI speaking rate, -10 to 10")
    ap.add_argument("--gap", type=float, default=GAP_S, help="silence between sentences")
    args = ap.parse_args()

    gap = np.zeros(int(SAMPLE_RATE * args.gap), dtype=np.int16)
    pieces: list[np.ndarray] = []

    with tempfile.TemporaryDirectory() as tmp:
        for i, line in enumerate(LECTURE, 1):
            part = Path(tmp) / f"line{i:02d}.wav"
            print(f"  [{i:2d}/{len(LECTURE)}] {line[:62]}...")
            if not speak(line, part, args.rate):
                print("  synthesis failed - Windows SAPI is required")
                return 1
            pieces.append(read_pcm(part))
            pieces.append(gap)

    audio = np.concatenate(pieces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())

    print(f"\nwrote {args.out}  ({audio.size / SAMPLE_RATE:.1f}s, {len(LECTURE)} sentences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
