"""KV-cache plumbing for Optimum "merged" encoder-decoder graphs.

Whisper and NLLB are both exported this way, and both failed in exactly the
same manner the first time they met a real graph: ONNX Runtime rejected the
call with dozens of missing ``past_key_values.*`` inputs. Rather than write
that twice, it lives here once.

A merged decoder carries two branches in one file, selected by the boolean
``use_cache_branch`` input:

* **first step** - ``use_cache_branch=False``, feed the whole prompt and
  zero-length caches. The graph computes self- and cross-attention KV and
  returns them as ``present.*``.
* **every step after** - ``use_cache_branch=True``, feed only the newest
  token and the previous step's ``present.*`` as ``past_key_values.*``.

Re-sending the full prefix each step and leaving the cache branch off also
produces correct text, which is why the bug is easy to miss. It is just
quadratic: on a 200-token caption that is the difference between a live
captioner and one that falls behind and never catches up.
"""

from __future__ import annotations

import numpy as np

PAST_PREFIX = "past_key_values"
PRESENT_PREFIX = "present"
CACHE_FLAG = "use_cache_branch"

# The axis that grows by one token per decode step. Layout is
# (batch, heads, sequence, head_dim) for every export we support.
SEQUENCE_AXIS = 2


class MergedDecoderCache:
    """Builds the cache tensors a merged decoder expects on each step."""

    def __init__(self, dec_inputs: dict, dec_outputs: list[str]):
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        self.past_names = [n for n in dec_inputs if n.startswith(PAST_PREFIX)]

    @property
    def active(self) -> bool:
        return CACHE_FLAG in self.dec_inputs and bool(self.past_names)

    def empty(self) -> dict[str, np.ndarray]:
        """Zero-length caches for the first step.

        Length 0, not 1. With ``use_cache_branch=False`` the graph
        concatenates these onto freshly computed keys, so a dummy row would
        become a phantom token that every subsequent step attends to -
        producing plausible but subtly wrong output, the worst failure mode
        available.
        """
        out: dict[str, np.ndarray] = {}
        for name in self.past_names:
            meta = self.dec_inputs[name]
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
            shape[SEQUENCE_AXIS] = 0
            out[name] = np.zeros(shape, dtype=np.float32)
        return out

    def collect(
        self,
        outputs: list,
        reuse_encoder: bool,
        previous: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Map this step's ``present.*`` outputs to next step's inputs.

        Cross-attention (``.encoder.``) KV depends only on the encoder output,
        so once the cache branch is live the graph stops recomputing it and
        echoes back whatever was passed in. Carrying the originals forward
        keeps that correct under either behaviour.
        """
        previous = previous or {}
        mapping: dict[str, np.ndarray] = {}

        for name, value in zip(self.dec_outputs, outputs, strict=False):
            if not name.startswith(PRESENT_PREFIX):
                continue
            target = name.replace(PRESENT_PREFIX, PAST_PREFIX, 1)
            if target not in self.dec_inputs:
                continue
            if reuse_encoder and ".encoder." in target and target in previous:
                mapping[target] = previous[target]
            else:
                mapping[target] = np.asarray(value, dtype=np.float32)

        return mapping

    @staticmethod
    def flag(first_pass: bool) -> np.ndarray:
        """``use_cache_branch`` is tensor(bool).

        Feeding it a float32 zero is what ORT rejected outright the first
        time, which at least fails loudly rather than silently.
        """
        return np.array([not first_pass], dtype=bool)
