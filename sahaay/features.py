"""Whisper's log-mel front end, in pure NumPy.

Deliberately not librosa or torchaudio: both drag in large dependency trees
with patchy Windows-ARM64 wheel coverage, and this is ~60 lines of maths
that has to match Whisper's preprocessing exactly or the encoder sees
out-of-distribution input and quietly produces fluent nonsense.

Reference: openai/whisper audio.py (n_fft=400, hop=160, slaney mel scale).
"""

from __future__ import annotations

import functools

import numpy as np

N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
SAMPLE_RATE = 16_000
N_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE      # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH           # 3000


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    # atleast_1d: a scalar would otherwise give a 0-d array, which cannot be
    # boolean-index-assigned below.
    hz = np.atleast_1d(np.asarray(hz, dtype=np.float64))
    f_min, f_sp = 0.0, 200.0 / 3
    mel = (hz - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    above = hz >= min_log_hz
    mel[above] = min_log_mel + np.log(hz[above] / min_log_hz) / logstep
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.atleast_1d(np.asarray(mel, dtype=np.float64))
    f_min, f_sp = 0.0, 200.0 / 3
    hz = f_min + f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    above = mel >= min_log_mel
    hz[above] = min_log_hz * np.exp(logstep * (mel[above] - min_log_mel))
    return hz


@functools.lru_cache(maxsize=4)
def mel_filterbank(n_mels: int = 80, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT) -> np.ndarray:
    """(n_mels, n_fft//2 + 1) triangular filterbank, Slaney-normalised."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_min = float(_hz_to_mel(0.0)[0])
    mel_max = float(_hz_to_mel(sample_rate / 2.0)[0])
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    ramps = hz_points[:, None] - fft_freqs[None, :]
    fdiff = np.diff(hz_points)

    lower = -ramps[:-2] / fdiff[:-1][:, None]
    upper = ramps[2:] / fdiff[1:][:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalisation: equal area per filter, not equal peak.
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


@functools.lru_cache(maxsize=2)
def _hann(n: int) -> np.ndarray:
    # Periodic Hann, matching torch.hann_window(periodic=True).
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)).astype(np.float32)


def log_mel_spectrogram(
    audio: np.ndarray, n_mels: int = 80, pad_to_frames: int | None = N_FRAMES
) -> np.ndarray:
    """Convert mono 16 kHz float audio into Whisper's (n_mels, frames) input."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if pad_to_frames is not None:
        target = pad_to_frames * HOP_LENGTH
        if audio.size < target:
            audio = np.pad(audio, (0, target - audio.size))
        else:
            audio = audio[:target]

    # Reflect-pad so frame centres line up with sample indices (center=True).
    pad = N_FFT // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")

    window = _hann(N_FFT)
    n_frames = 1 + (padded.size - N_FFT) // HOP_LENGTH
    # Strided view avoids materialising a copy per frame.
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, N_FFT),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
    )
    spec = np.fft.rfft(frames * window, n=N_FFT, axis=-1)
    magnitudes = (spec.real**2 + spec.imag**2).astype(np.float32)

    # Whisper drops the final frame (it has no matching audio).
    magnitudes = magnitudes[:-1, :].T

    mel = mel_filterbank(n_mels) @ magnitudes
    log_spec = np.log10(np.maximum(mel, 1e-10))
    # Clamp the dynamic range to 8 dec, then scale to roughly [-1, 1].
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size > length:
        return audio[:length]
    if audio.size < length:
        return np.pad(audio, (0, length - audio.size))
    return audio
