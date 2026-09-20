"""Audio capture: microphone and WASAPI loopback, mixed to 16 kHz mono.

Loopback matters more than the mic for the target use case. A student in an
online class needs the *lecturer's* audio, which is coming out of their own
speakers - WASAPI loopback captures that without a virtual cable or any
driver install, which is what makes one-click deployment possible.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AudioConfig

log = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool


class AudioSource:
    """Base class so the mock source and the real one are interchangeable."""

    def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self, timeout: float = 1.0) -> np.ndarray | None:  # pragma: no cover
        raise NotImplementedError


class WasapiSource(AudioSource):
    """Captures speaker output and/or microphone via PyAudioWPatch.

    Both streams are resampled to 16 kHz mono and summed. Summing rather than
    picking one means a hybrid classroom - a lecturer in the room plus a
    remote participant on the call - produces a single coherent transcript.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._streams: list = []
        self._pa = None
        self._running = False
        # Per-source leftovers from resampling, keyed by stream index.
        self._resample_tail: dict[int, np.ndarray] = {}

    # -- device discovery --------------------------------------------------

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        try:
            import pyaudiowpatch as pyaudio  # type: ignore
        except ImportError:
            return []
        out: list[DeviceInfo] = []
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if int(d.get("maxInputChannels", 0)) <= 0:
                    continue
                out.append(
                    DeviceInfo(
                        index=i,
                        name=str(d.get("name", f"device {i}")),
                        channels=int(d["maxInputChannels"]),
                        sample_rate=int(d.get("defaultSampleRate", 48000)),
                        is_loopback=bool(d.get("isLoopbackDevice", False)),
                    )
                )
        finally:
            pa.terminate()
        return out

    def _open(self, pa, device_info: dict, tag: int) -> None:
        native_rate = int(device_info["defaultSampleRate"])
        channels = int(device_info["maxInputChannels"])
        # 32 ms of audio per callback keeps VAD frames aligned and latency low.
        frames = max(256, int(native_rate * 0.032))

        def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
            try:
                samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                samples = self._to_16k(samples, native_rate, tag)
                if samples.size:
                    self._q.put_nowait(samples)
            except queue.Full:
                # Dropping a frame is better than stalling the audio driver
                # callback, which would glitch the user's own playback.
                log.debug("audio queue full; dropped frame")
            except Exception as exc:  # noqa: BLE001
                log.warning("audio callback error: %s", exc)
            import pyaudiowpatch as pyaudio  # type: ignore

            return (None, pyaudio.paContinue)

        stream = pa.open(
            format=8,  # pyaudio.paInt16
            channels=channels,
            rate=native_rate,
            input=True,
            input_device_index=int(device_info["index"]),
            frames_per_buffer=frames,
            stream_callback=callback,
        )
        self._streams.append(stream)

    def _to_16k(self, samples: np.ndarray, src_rate: int, tag: int) -> np.ndarray:
        """Linear-interpolation resample to 16 kHz.

        Whisper's mel front end is tolerant enough that polyphase filtering
        buys no measurable WER here, and linear keeps scipy out of the
        dependency list - which matters for ARM64 wheel availability.
        """
        if src_rate == self.cfg.sample_rate:
            return samples.astype(np.float32)

        tail = self._resample_tail.get(tag)
        if tail is not None and tail.size:
            samples = np.concatenate([tail, samples])

        ratio = self.cfg.sample_rate / src_rate
        n_out = int(samples.size * ratio)
        if n_out < 1:
            self._resample_tail[tag] = samples
            return np.empty(0, dtype=np.float32)

        # Keep the remainder so consecutive buffers stay phase-continuous.
        consumed = int(n_out / ratio)
        self._resample_tail[tag] = samples[consumed:].copy()

        idx = np.arange(n_out, dtype=np.float32) / ratio
        lo = np.floor(idx).astype(np.int32)
        hi = np.minimum(lo + 1, samples.size - 1)
        frac = idx - lo
        return ((1.0 - frac) * samples[lo] + frac * samples[hi]).astype(np.float32)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        try:
            import pyaudiowpatch as pyaudio  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyaudiowpatch is required for audio capture on Windows.\n"
                "  pip install pyaudiowpatch\n"
                "Or run with --mock to exercise the pipeline without audio."
            ) from exc

        self._pa = pyaudio.PyAudio()
        opened = 0

        if self.cfg.capture_loopback:
            try:
                default_speakers = self._default_loopback(self._pa, pyaudio)
                if default_speakers:
                    self._open(self._pa, default_speakers, tag=0)
                    opened += 1
                    log.info("capturing system audio: %s", default_speakers["name"])
            except Exception as exc:  # noqa: BLE001
                log.warning("loopback capture unavailable: %s", exc)

        if self.cfg.capture_microphone:
            try:
                mic = self._pa.get_default_input_device_info()
                self._open(self._pa, dict(mic), tag=1)
                opened += 1
                log.info("capturing microphone: %s", mic["name"])
            except Exception as exc:  # noqa: BLE001
                log.warning("microphone capture unavailable: %s", exc)

        if not opened:
            raise RuntimeError("No audio source could be opened (tried loopback and microphone).")

        self._running = True
        for s in self._streams:
            s.start_stream()

    @staticmethod
    def _default_loopback(pa, pyaudio) -> dict | None:  # noqa: ANN001
        """Find the loopback device matching the current default speakers."""
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return None
        default_out = pa.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
        for loopback in pa.get_loopback_device_info_generator():
            if default_out["name"] in loopback["name"]:
                return dict(loopback)
        return None

    def stop(self) -> None:
        self._running = False
        for s in self._streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:  # noqa: BLE001, S110
                pass
        self._streams.clear()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class MockSource(AudioSource):
    """Silence generator used by ``--mock``.

    The mock pipeline injects scripted transcript text downstream, so this
    only has to keep the timing realistic.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._running = False
        self._stop = threading.Event()

    def start(self) -> None:
        self._running = True
        self._stop.clear()

    def stop(self) -> None:
        self._running = False
        self._stop.set()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if not self._running:
            return None
        self._stop.wait(0.032)
        n = int(self.cfg.sample_rate * 0.032)
        return np.zeros(n, dtype=np.float32)


def load_wav(path: Path, target_rate: int) -> np.ndarray:
    """Read a WAV file to float32 mono at ``target_rate``.

    Deliberately stdlib-only. Pulling in soundfile or librosa to read a
    16-bit PCM file would add a compiled dependency to the install for
    something ``wave`` already does.
    """
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 1:  # unsigned, midpoint 128
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width * 8}-bit")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if rate != target_rate:
        n = int(round(data.size * target_rate / rate))
        data = np.interp(
            np.linspace(0.0, data.size - 1, n, dtype=np.float64),
            np.arange(data.size, dtype=np.float64),
            data,
        ).astype(np.float32)

    return np.ascontiguousarray(data, dtype=np.float32)


class FileSource(AudioSource):
    """Play a WAV file through the pipeline as though it were live audio.

    This exists because the pipeline is otherwise only reachable through a
    sound card: ``WasapiSource`` captures whatever the speakers are playing,
    which cannot be scripted, and ``MockSource`` emits silence and bypasses
    the VAD entirely. Neither lets anyone reproduce a real run.

    Chunks are paced against the wall clock rather than returned as fast as
    they can be read, because the pipeline's behaviour under load is the
    interesting part - handing it an hour of audio instantly would measure
    throughput, not the live path. ``rate`` scales that pacing.

    A tail of silence is appended so the segmenter's pause detector flushes
    the final phrase; without it the last sentence of every file is lost.
    """

    def __init__(self, cfg: AudioConfig, path: Path, rate: float = 1.0, tail_s: float = 1.2):
        self.cfg = cfg
        self.path = Path(path)
        self.rate = max(0.0, rate)
        samples = load_wav(self.path, cfg.sample_rate)
        tail = np.zeros(int(cfg.sample_rate * tail_s), dtype=np.float32)
        self._samples = np.concatenate([samples, tail])
        self.duration_s = samples.size / cfg.sample_rate
        self._pos = 0
        self._running = False
        self._stop = threading.Event()
        self._t0 = 0.0
        self.exhausted = False

    def start(self) -> None:
        self._pos = 0
        self.exhausted = False
        self._running = True
        self._stop.clear()
        self._t0 = time.monotonic()

    def stop(self) -> None:
        self._running = False
        self._stop.set()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if not self._running:
            return None

        if self._pos >= self._samples.size:
            # Keep behaving like a live source that has gone quiet rather
            # than tearing down: the caller decides when the session ends.
            self.exhausted = True
            self._stop.wait(min(timeout, 0.05))
            return None

        n = int(self.cfg.sample_rate * 0.032)
        chunk = self._samples[self._pos : self._pos + n]
        self._pos += chunk.size

        if self.rate > 0:
            due = self._t0 + (self._pos / self.cfg.sample_rate) / self.rate
            delay = due - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)

        return chunk


def create_source(
    cfg: AudioConfig,
    mock: bool = False,
    audio_file: Path | None = None,
    rate: float = 1.0,
) -> AudioSource:
    if audio_file is not None:
        return FileSource(cfg, audio_file, rate=rate)
    return MockSource(cfg) if mock else WasapiSource(cfg)
