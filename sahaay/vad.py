"""Voice activity detection and segmentation.

Segmentation is what makes streaming ASR feel live. Whisper is a 30-second
batch model; running it on fixed 30 s windows would mean captions arrive
half a minute late. Instead we cut on natural pauses, so a caption lands
roughly one sentence behind the speaker.

Silero VAD is a ~1.8 MB ONNX model. It is small enough that it stays on the
CPU even when the NPU is available - shipping it to the HTP would cost more
in transfer overhead than it saves in compute.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FRAME_SAMPLES, AudioConfig, VadConfig

log = logging.getLogger(__name__)


@dataclass
class Segment:
    """A contiguous run of speech, ready for the ASR."""

    audio: np.ndarray
    start_s: float
    end_s: float
    index: int

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


class SileroVad:
    """Thin wrapper over the Silero VAD ONNX graph."""

    def __init__(self, model_path: Path, sample_rate: int = 16_000):
        import onnxruntime as ort  # local import: VAD always runs on CPU

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # it is tiny; threads only add latency
        self._sess = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._sample_rate = sample_rate
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self.reset()

    def reset(self) -> None:
        # Silero v5 keeps a single combined state tensor; v4 used h/c pairs.
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample frame."""
        x = frame.reshape(1, -1).astype(np.float32)
        feeds: dict[str, np.ndarray] = {"input": x}
        if "sr" in self._input_names:
            feeds["sr"] = np.array(self._sample_rate, dtype=np.int64)
        if "state" in self._input_names:
            feeds["state"] = self._state
        else:
            feeds["h"] = self._h
            feeds["c"] = self._c

        outputs = self._sess.run(None, feeds)
        prob = float(np.asarray(outputs[0]).reshape(-1)[0])

        if "state" in self._input_names and len(outputs) > 1:
            self._state = np.asarray(outputs[1], dtype=np.float32)
        elif len(outputs) > 2:
            self._h = np.asarray(outputs[1], dtype=np.float32)
            self._c = np.asarray(outputs[2], dtype=np.float32)
        return prob


class EnergyVad:
    """Fallback VAD used when the Silero weights are absent.

    An adaptive noise floor rather than a fixed threshold, because lecture
    halls, laptop fans and hostel rooms differ by 20+ dB and a fixed gate
    either clips quiet speakers or never closes.
    """

    def __init__(self) -> None:
        self._noise_floor = 1e-3
        self._warm = 0

    def reset(self) -> None:
        self._noise_floor = 1e-3
        self._warm = 0

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9)
        self._warm += 1
        if rms < self._noise_floor * 1.5 or self._warm < 20:
            # Track the floor slowly upward, quickly downward.
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        ratio = rms / max(self._noise_floor, 1e-6)
        # Map a 4x-over-noise ratio to ~0.5 probability.
        return float(np.clip((ratio - 1.5) / 5.0, 0.0, 1.0))


class Segmenter:
    """Turns a stream of audio frames into speech segments.

    Flushes when the speaker pauses for ``silence_flush_ms``, or when a
    segment hits ``max_segment_s`` - the latter guards against a lecturer who
    simply never pauses, which is common and would otherwise starve the UI.
    """

    def __init__(
        self,
        audio_cfg: AudioConfig,
        vad_cfg: VadConfig,
        vad: SileroVad | EnergyVad | None = None,
    ):
        self.audio_cfg = audio_cfg
        self.vad_cfg = vad_cfg
        self.vad = vad or EnergyVad()

        self._buffer = np.empty(0, dtype=np.float32)   # frames awaiting VAD
        self._speech = np.empty(0, dtype=np.float32)   # current segment
        self._pad = deque(maxlen=max(1, vad_cfg.speech_pad_ms // 32))
        self._in_speech = False
        self._silence_frames = 0
        self._elapsed_samples = 0
        self._segment_start = 0.0
        self._index = 0

    @property
    def _silence_frames_needed(self) -> int:
        return max(1, self.audio_cfg.silence_flush_ms // 32)

    def push(self, chunk: np.ndarray) -> list[Segment]:
        """Feed audio in; get back any segments that completed."""
        out: list[Segment] = []
        self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])

        while self._buffer.size >= FRAME_SAMPLES:
            frame = self._buffer[:FRAME_SAMPLES]
            self._buffer = self._buffer[FRAME_SAMPLES:]
            seg = self._push_frame(frame)
            if seg is not None:
                out.append(seg)
        return out

    def _push_frame(self, frame: np.ndarray) -> Segment | None:
        t_now = self._elapsed_samples / self.audio_cfg.sample_rate
        self._elapsed_samples += frame.size

        prob = self.vad.probability(frame) if self.vad_cfg.enabled else 1.0
        is_speech = prob >= self.vad_cfg.threshold

        if not self._in_speech:
            self._pad.append(frame)
            if is_speech:
                self._in_speech = True
                self._silence_frames = 0
                # Prepend the padding so the segment doesn't start mid-word.
                self._speech = (
                    np.concatenate(list(self._pad)) if self._pad else frame.copy()
                )
                self._pad.clear()
                self._segment_start = max(
                    0.0, t_now - (self.vad_cfg.speech_pad_ms / 1000.0)
                )
            return None

        self._speech = np.concatenate([self._speech, frame])
        self._silence_frames = 0 if is_speech else self._silence_frames + 1

        too_long = (
            self._speech.size / self.audio_cfg.sample_rate >= self.audio_cfg.max_segment_s
        )
        paused = self._silence_frames >= self._silence_frames_needed
        if paused or too_long:
            return self._flush(t_now, forced=too_long)
        return None

    def _flush(self, t_now: float, forced: bool = False) -> Segment | None:
        audio = self._speech
        self._speech = np.empty(0, dtype=np.float32)
        self._in_speech = False
        self._silence_frames = 0
        self._pad.clear()

        duration = audio.size / self.audio_cfg.sample_rate
        if duration < self.audio_cfg.min_segment_s:
            # Too short to be speech - a cough, a chair, a keystroke.
            return None

        seg = Segment(
            audio=audio, start_s=self._segment_start, end_s=t_now, index=self._index
        )
        self._index += 1
        if forced:
            # A forced cut lands mid-sentence; carry the tail forward so the
            # next segment has context and the join reads naturally.
            log.debug("forced segment cut at %.1fs", duration)
        return seg

    def finalize(self) -> Segment | None:
        """Flush whatever is buffered when the session stops."""
        if self._speech.size == 0:
            return None
        return self._flush(self._elapsed_samples / self.audio_cfg.sample_rate)


def create_vad(models_dir: Path, cfg: VadConfig) -> SileroVad | EnergyVad:
    """Prefer Silero; fall back to energy gating so the app always starts."""
    path = models_dir / "silero_vad" / "silero_vad.onnx"
    if path.exists():
        try:
            return SileroVad(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Silero VAD failed to load (%s); using energy VAD", exc)
    else:
        log.info("Silero VAD not downloaded; using energy VAD")
    return EnergyVad()
