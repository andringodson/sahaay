"""The log-mel front end.

Worth real tests: if this is subtly wrong the encoder still produces fluent
text, it is just wrong text, and nothing downstream will flag it.
"""

import numpy as np
import pytest

from sahaay.features import (
    N_FRAMES,
    N_SAMPLES,
    log_mel_spectrogram,
    mel_filterbank,
    pad_or_trim,
)

SR = 16_000


def tone(freq: float, seconds: float = 3.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestFilterbank:
    def test_shape_matches_rfft_bins(self):
        fb = mel_filterbank(80)
        assert fb.shape == (80, 201)  # n_fft // 2 + 1

    def test_every_filter_has_energy(self):
        # An all-zero filter row means a mel band that can never fire, which
        # silently deletes part of the spectrum.
        assert np.all(mel_filterbank(80).sum(axis=1) > 0)

    def test_filters_are_ordered_low_to_high(self):
        fb = mel_filterbank(80)
        peaks = fb.argmax(axis=1)
        assert np.all(np.diff(peaks) >= 0)

    def test_large_v3_uses_128_bins(self):
        assert mel_filterbank(128).shape == (128, 201)


class TestLogMel:
    def test_output_shape_is_whisper_input_shape(self):
        assert log_mel_spectrogram(tone(440)).shape == (80, N_FRAMES)

    def test_dtype_is_float32(self):
        # float64 here would silently double the bytes pushed to the NPU.
        assert log_mel_spectrogram(tone(440)).dtype == np.float32

    def test_normalised_range(self):
        mel = log_mel_spectrogram(tone(440))
        # Whisper clamps to 8 decades then maps to roughly [-1, 1].
        assert -1.1 <= float(mel.min()) <= 0.5
        assert 0.5 <= float(mel.max()) <= 1.5

    @pytest.mark.parametrize(
        "freq,low,high",
        [(200, 0, 12), (440, 5, 20), (2000, 30, 55), (6000, 60, 79)],
    )
    def test_tone_lands_in_expected_mel_band(self, freq, low, high):
        # A sine at f must peak in the mel bins covering f. This is the test
        # that actually catches a wrong mel scale.
        mel = log_mel_spectrogram(tone(freq))
        peak = int(mel[:, 50].argmax())
        assert low <= peak <= high, f"{freq} Hz peaked at bin {peak}"

    def test_short_audio_is_padded_not_rejected(self):
        assert log_mel_spectrogram(tone(440, seconds=0.5)).shape == (80, N_FRAMES)

    def test_long_audio_is_trimmed_to_30s(self):
        assert log_mel_spectrogram(tone(440, seconds=60)).shape == (80, N_FRAMES)

    def test_silence_does_not_produce_nan(self):
        mel = log_mel_spectrogram(np.zeros(SR * 2, dtype=np.float32))
        assert np.isfinite(mel).all()

    def test_deterministic(self):
        a = tone(440)
        assert np.array_equal(log_mel_spectrogram(a), log_mel_spectrogram(a))


class TestPadOrTrim:
    def test_pads_short(self):
        assert pad_or_trim(np.zeros(1000, dtype=np.float32)).size == N_SAMPLES

    def test_trims_long(self):
        assert pad_or_trim(np.zeros(N_SAMPLES * 2, dtype=np.float32)).size == N_SAMPLES

    def test_exact_length_untouched(self):
        a = np.ones(N_SAMPLES, dtype=np.float32)
        assert np.array_equal(pad_or_trim(a), a)
