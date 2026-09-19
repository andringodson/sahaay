"""Merged-decoder KV cache.

Every assertion here corresponds to a failure observed against a real ONNX
graph, not a hypothetical. Whisper and NLLB both broke in the same three
ways the first time they met actual weights.
"""

import numpy as np
import pytest

from sahaay.kvcache import CACHE_FLAG, MergedDecoderCache


class FakeMeta:
    def __init__(self, shape, type_="tensor(float)"):
        self.shape = shape
        self.type = type_


def make_cache(layers: int = 2) -> MergedDecoderCache:
    inputs = {"input_ids": FakeMeta(["b", "seq"], "tensor(int64)"),
              "encoder_hidden_states": FakeMeta(["b", "enc", 384]),
              CACHE_FLAG: FakeMeta([1], "tensor(bool)")}
    outputs = ["logits"]
    for i in range(layers):
        for side in ("decoder", "encoder"):
            for kv in ("key", "value"):
                inputs[f"past_key_values.{i}.{side}.{kv}"] = FakeMeta(["b", 6, "past", 64])
                outputs.append(f"present.{i}.{side}.{kv}")
    return MergedDecoderCache(inputs, outputs)


class TestDetection:
    def test_active_when_the_graph_is_merged(self):
        assert make_cache().active is True

    def test_inactive_without_the_flag(self):
        c = MergedDecoderCache({"input_ids": FakeMeta(["b", "s"])}, ["logits"])
        assert c.active is False

    def test_finds_every_past_input(self):
        assert len(make_cache(layers=3).past_names) == 12


class TestEmptyCache:
    def test_sequence_axis_is_zero_length(self):
        # Length 1 would inject a phantom token that the decoder attends to
        # for the rest of the sequence - plausible but wrong output, which
        # is the worst failure mode available.
        for tensor in make_cache().empty().values():
            assert tensor.shape[2] == 0

    def test_covers_all_past_inputs(self):
        c = make_cache(layers=4)
        assert set(c.empty()) == set(c.past_names)

    def test_dtype_is_float32(self):
        assert all(t.dtype == np.float32 for t in make_cache().empty().values())


class TestFlag:
    def test_is_bool_not_float(self):
        # Feeding float32 here is what ORT rejected outright:
        # "Unexpected input data type. Actual: (tensor(float)), expected: (tensor(bool))"
        assert MergedDecoderCache.flag(True).dtype == np.bool_

    @pytest.mark.parametrize("first,expected", [(True, False), (False, True)])
    def test_semantics_are_inverted(self, first, expected):
        # The flag says "use the cache", which is the opposite of "this is
        # the first pass". Getting this backwards silently halves quality.
        assert bool(MergedDecoderCache.flag(first)[0]) is expected


class TestCollect:
    def outputs(self, cache, fill=1.0):
        out = [np.zeros((1, 4, 100), np.float32)]  # logits
        for _ in range(len(cache.past_names)):
            out.append(np.full((1, 6, 3, 64), fill, np.float32))
        return out

    def test_maps_present_onto_past(self):
        c = make_cache()
        mapped = c.collect(self.outputs(c), reuse_encoder=False)
        assert set(mapped) == set(c.past_names)

    def test_logits_are_not_treated_as_cache(self):
        c = make_cache()
        assert "logits" not in c.collect(self.outputs(c), reuse_encoder=False)

    def test_encoder_cache_is_carried_forward(self):
        # Cross-attention KV depends only on the encoder output. Once the
        # cache branch is live the graph echoes back whatever it was given,
        # so the originals must be preserved rather than overwritten.
        c = make_cache()
        first = c.collect(self.outputs(c, fill=1.0), reuse_encoder=False)
        second = c.collect(self.outputs(c, fill=9.0), reuse_encoder=True, previous=first)

        enc = [k for k in second if ".encoder." in k]
        dec = [k for k in second if ".decoder." in k]
        assert enc and dec
        assert all(second[k][0, 0, 0, 0] == 1.0 for k in enc), "encoder cache was clobbered"
        assert all(second[k][0, 0, 0, 0] == 9.0 for k in dec), "decoder cache went stale"
