# Transcription accuracy

Measured 20 Sep 2026 by `scripts/accuracy.py` on **CPU**, model `whisper_small_portable`.

## Results

| Set | Utterances | WER | Technical terms kept |
|---|---:|---:|---:|
| english | 4 | **0.0%** | 8/8 |
| code-mixed | 3 | **63.6%** | 6/6 |
| **all** | 7 | **23.7%** | 14/14 |

## Per utterance

| Ref | Hypothesis | WER |
|---|---|---:|
| So today we will start with eigenvalues and eigenvectors. | So today we will start with eigenvalues and eigenvectors. | 0.0% |
| The determinant of matrix A must be zero for a non trivial solution. | The determinant of matrix A must be zero for a non-trivial solution. | 0.0% |
| This is called the characteristic equation of the matrix. | This is called the characteristic equation of the matrix. | 0.0% |
| Symmetric matrices always have real eigenvalues. | Symmetric matrices always have real eigenvalues. | 0.0% |
| Aaj hum eigenvalues aur eigenvectors padhenge. | Aajhum eigenvalues or eigenvectors pad hinge. | 83.3% |
| Matrix A ka determinant zero hoga tabhi non trivial solution milega. | Matrix ACA determinant 0 Haugetab high non-trivial solution Milaga | 45.5% |
| Yeh characteristic equation kehlata hai. | Yeah characteristic equation K-lada Hi. | 80.0% |

## What this measures, and what it does not

**The audio is synthesised by Windows SAPI, not spoken by people.** That
makes ground truth exact and lets anyone reproduce this on a Windows
machine with no dataset to download. It is a fair test of the decoding
path - mel front end, KV cache, language selection, detokenisation.

It is **not** a substitute for a real corpus. Synthetic speech has no
accent, no room reverb, no crosstalk and no disfluency, so these numbers
are a floor rather than a forecast. Expect materially worse on a real
lecture recording.

For the code-mixed set the caveat is stronger: only an **en-US** voice is
installed, so romanised Hindi is read with English phonetics. That tests
whether the pipeline carries technical terms through non-English word
sequences. It tells you nothing about how the model handles an actual
Hindi speaker, and the WER for that set should be read as a lower bound on
difficulty, not an estimate of field accuracy.

**Technical terms kept** is scored separately because it is the metric the
product actually depends on. A student can work around a garbled function
word; a garbled `eigenvalue` is the one that breaks the link to the
textbook.

Numbers are normalised before scoring ("0" and "zero" count as the same
word), since that is a formatting difference rather than a recognition
error.
