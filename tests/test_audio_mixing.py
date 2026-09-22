"""Two capture streams have to be summed, not interleaved.

They used to share one queue, so read() handed the segmenter alternating
32 ms chunks from the loopback and the microphone rather than their sum -
lecture audio chopped with room audio. It also meant every second of audio
was counted twice, so the app's own real-time factor read about twice as
good as it was.

Nothing caught it because every test and every benchmark ran with a single
source. It surfaced only when a tone was played through the speakers and
12.4 s of samples arrived during a 6 s listen.

These tests drive the mixer directly, so they need no sound card.
"""

from __future__ import annotations

import numpy as np
import pytest

from sahaay.audio import WasapiSource
from sahaay.config import AudioConfig

BLOCK = WasapiSource.BLOCK


@pytest.fixture
def source():
    return WasapiSource(AudioConfig())


def test_one_stream_comes_back_unchanged(source):
    samples = np.full(BLOCK, 0.25, dtype=np.float32)
    source._push(0, samples)

    out = source.read(timeout=0.5)
    assert out is not None
    assert out.size == BLOCK
    assert np.allclose(out, 0.25)


def test_two_streams_are_summed_not_interleaved(source):
    """The whole point. A block must contain both sources at once."""
    source._push(0, np.full(BLOCK, 0.25, dtype=np.float32))
    source._push(1, np.full(BLOCK, 0.10, dtype=np.float32))

    out = source.read(timeout=0.5)
    assert out is not None
    assert out.size == BLOCK
    # Interleaving would give a block that is all 0.25 or all 0.10.
    assert np.allclose(out, 0.35), f"expected the sum, got {out[:4]}"


def test_a_second_read_does_not_replay_the_first_block(source):
    source._push(0, np.concatenate([
        np.full(BLOCK, 0.2, dtype=np.float32),
        np.full(BLOCK, 0.4, dtype=np.float32),
    ]))

    first = source.read(timeout=0.5)
    second = source.read(timeout=0.5)
    assert np.allclose(first, 0.2)
    assert np.allclose(second, 0.4)


def test_audio_is_not_counted_twice(source):
    """Two streams delivering one second must read back as one second.

    This is the metrics half of the bug: audio_seconds drives the real-time
    factor, and counting each second twice halves the reported RTF.
    """
    rate = AudioConfig().sample_rate
    one_second = np.zeros(rate, dtype=np.float32)
    source._push(0, one_second.copy())
    source._push(1, one_second.copy())

    total = 0
    while True:
        block = source.read(timeout=0.05)
        if block is None:
            break
        total += block.size

    assert total == pytest.approx(rate, rel=0.02), (
        f"{total / rate:.2f}s came back from 1s delivered on two streams"
    )


def test_a_silent_stream_does_not_stall_the_mix(source):
    """A muted or unplugged device must not hold the lecture up."""
    source._push(0, np.full(BLOCK, 0.3, dtype=np.float32))
    # Stream 1 registered once, long ago, and has gone quiet.
    source._pending[1] = np.empty(0, dtype=np.float32)
    source._last_seen[1] = 0.0

    out = source.read(timeout=0.5)
    assert out is not None
    assert np.allclose(out, 0.3)


def test_the_sum_is_clipped_rather_than_wrapped(source):
    """Two loud sources sum past full scale; wrap-around would reach the mel."""
    source._push(0, np.full(BLOCK, 0.9, dtype=np.float32))
    source._push(1, np.full(BLOCK, 0.9, dtype=np.float32))

    out = source.read(timeout=0.5)
    assert out is not None
    assert float(out.max()) <= 1.0
    assert np.allclose(out, 1.0)


def test_read_returns_none_when_nothing_has_arrived(source):
    assert source.read(timeout=0.05) is None


def test_a_buffer_nobody_drains_stays_bounded(source):
    """A stream running ahead must not grow without limit for a whole lecture."""
    rate = AudioConfig().sample_rate
    for _ in range(10):
        source._push(0, np.zeros(rate, dtype=np.float32))

    assert source._pending[0].size <= rate, "per-stream buffer grew past its cap"
