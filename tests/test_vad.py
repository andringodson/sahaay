"""Segmentation behaviour.

The segmenter decides when a caption appears. Getting it wrong is not a
crash, it is a product that feels broken - captions arriving 30 seconds late,
or one word at a time.
"""

import numpy as np

from sahaay.config import AudioConfig, VadConfig
from sahaay.vad import EnergyVad, Segmenter

SR = 16_000


def speech(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    signal = sum(amplitude * np.sin(2 * np.pi * f * t) for f in (150, 700, 1800))
    return (signal * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    # Never pure zeros: a real room has a noise floor, and a VAD that only
    # works against digital silence does not work at all.
    return (np.random.randn(int(seconds * SR)) * 1e-4).astype(np.float32)


def make(**overrides) -> Segmenter:
    audio = AudioConfig(**{k: v for k, v in overrides.items() if hasattr(AudioConfig, k)})
    return Segmenter(audio, VadConfig(), EnergyVad())


class TestEnergyVad:
    def test_silence_scores_low(self):
        vad = EnergyVad()
        frame = np.random.randn(512).astype(np.float32) * 1e-4
        for _ in range(40):
            vad.probability(frame)
        assert vad.probability(frame) < 0.5

    def test_speech_scores_high(self):
        vad = EnergyVad()
        quiet = np.random.randn(512).astype(np.float32) * 1e-4
        for _ in range(40):
            vad.probability(quiet)
        loud = speech(0.032)[:512]
        assert vad.probability(loud) > 0.5

    def test_adapts_to_a_noisy_room(self):
        # A loud fan must not read as continuous speech.
        vad = EnergyVad()
        noisy = np.random.randn(512).astype(np.float32) * 0.02
        for _ in range(80):
            vad.probability(noisy)
        assert vad.probability(noisy) < 0.5


class TestSegmenter:
    def test_silence_alone_yields_nothing(self):
        seg = make()
        assert seg.push(silence(3.0)) == []

    def test_speech_then_pause_emits_one_segment(self):
        seg = make()
        seg.push(silence(0.5))
        seg.push(speech(2.5))
        out = seg.push(silence(1.5))
        assert len(out) == 1
        assert out[0].duration >= 1.0

    def test_short_blip_is_discarded(self):
        # A cough or a chair scrape is not a caption.
        seg = make()
        seg.push(silence(0.5))
        seg.push(speech(0.2))
        assert seg.push(silence(1.5)) == []

    def test_continuous_speech_is_force_cut(self):
        # A lecturer who never pauses must not starve the UI.
        seg = make()
        seg.push(silence(0.3))
        out = seg.push(speech(30.0))
        assert len(out) >= 2
        assert all(s.duration <= AudioConfig().max_segment_s + 0.5 for s in out)

    def test_segments_are_indexed_in_order(self):
        seg = make()
        collected = []
        for _ in range(3):
            seg.push(silence(0.4))
            collected += seg.push(speech(2.0))
            collected += seg.push(silence(1.2))
        assert [s.index for s in collected] == list(range(len(collected)))

    def test_timestamps_advance(self):
        seg = make()
        collected = []
        for _ in range(3):
            seg.push(silence(0.4))
            collected += seg.push(speech(2.0))
            collected += seg.push(silence(1.2))
        starts = [s.start_s for s in collected]
        assert starts == sorted(starts)

    def test_finalize_flushes_a_sentence_in_progress(self):
        # Pressing Stop mid-sentence must not lose the sentence.
        seg = make()
        seg.push(silence(0.4))
        seg.push(speech(3.0))
        tail = seg.finalize()
        assert tail is not None and tail.duration >= 1.0

    def test_finalize_on_empty_returns_none(self):
        assert make().finalize() is None

    def test_arbitrary_chunk_sizes(self):
        # Audio callbacks do not deliver neat frame multiples.
        seg = make()
        audio = np.concatenate([silence(0.4), speech(2.5), silence(1.5)])
        out = []
        i = 0
        for size in (137, 4001, 512, 9999, 71):
            while i < audio.size:
                out += seg.push(audio[i : i + size])
                i += size
        assert len(out) >= 1


class TestForcedCut:
    """A lecturer who never pauses for 700 ms is normal, not an edge case.

    The 12 s cap used to cut wherever the clock landed, usually inside a
    word, and a comment claimed the tail was carried forward when it was
    not - so the split word was lost from both captions. The cut now moves
    to the nearest gap between words, and the remainder starts the next
    segment.
    """

    def test_the_cut_lands_in_the_gap_between_words(self):
        # 11.2 s of speech, a 60 ms breath, then more speech: the only quiet
        # place near the 12 s cap is the breath.
        seg = make()
        seg.push(silence(0.3))
        audio = np.concatenate([speech(11.2), silence(0.06), speech(8.0)])
        out = seg.push(audio)
        assert out, "a 19 s run of speech was never cut"

        first = out[0]
        # The pre-roll pads a little before the speech; the cut should sit at
        # the breath, i.e. about 11.2 s of speech in, not at the 12 s cap.
        assert 10.9 <= first.duration <= 11.8, (
            f"cut at {first.duration:.2f}s - expected the breath near 11.2s"
        )

    def test_nothing_is_lost_or_repeated_across_a_forced_cut(self):
        seg = make()
        seg.push(silence(0.3))
        body = speech(26.0)
        out = seg.push(body)
        tail = seg.finalize()
        segments = out + ([tail] if tail is not None else [])

        emitted = sum(s.audio.size for s in segments)
        # Everything spoken comes back out once: the pre-roll adds a few
        # frames of room tone before the first word, never less than the
        # speech itself and never a second copy of any of it.
        assert body.size <= emitted <= body.size + int(0.3 * SR), (
            f"{emitted / SR:.2f}s came out of {body.size / SR:.2f}s of speech"
        )

    def test_segment_times_join_up_after_a_cut(self):
        seg = make()
        seg.push(silence(0.3))
        out = seg.push(speech(26.0))
        assert len(out) >= 2
        for a, b in zip(out, out[1:], strict=False):
            assert abs(b.start_s - a.end_s) < 0.05, (
                f"gap or overlap between segments: {a.end_s:.2f} -> {b.start_s:.2f}"
            )
