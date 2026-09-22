"""Is sound actually reaching Sahaay? Prove it before the demo, not during it.

``--selftest`` reports "audio capture: N input devices, M loopback". That is
a device *listing*. It stays green when the loopback device is the wrong one,
when the output is muted, when nothing is playing, and when the capture
stream opens and returns silence forever. Every one of those looks identical
from the UI: you press Start and no captions appear.

So this opens the real capture path and measures what comes out of it.

    python scripts/check_audio.py                 # listen for 5 s
    python scripts/check_audio.py --play          # play a tone and listen
    python scripts/check_audio.py --seconds 10

``--play`` is the one to run. It sends a tone to the default output device
and checks the capture path hears it, which tests the loop end to end without
needing a video open. If that passes, a lecture will caption.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahaay.config import load_config  # noqa: E402

# Below this the capture is indistinguishable from a muted stream. Chosen
# against measured silence, which sits around 1e-5 RMS on a real device.
SILENCE_RMS = 1e-4
QUIET_RMS = 2e-3


def write_tone(path: Path, seconds: float, rate: int = 44_100) -> None:
    """A warbling tone - easy for a person to hear, easy to measure."""
    frames = []
    for i in range(int(seconds * rate)):
        t = i / rate
        freq = 440.0 + 120.0 * math.sin(2 * math.pi * 0.8 * t)
        value = 0.35 * math.sin(2 * math.pi * freq * t)
        frames.append(struct.pack("<h", int(value * 32767)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(frames))


def play_async(path: Path) -> bool:
    try:
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  could not play the tone ({exc}); play something yourself instead")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=5.0, help="how long to listen")
    ap.add_argument("--play", action="store_true", help="play a test tone while listening")
    args = ap.parse_args()

    cfg = load_config()

    from sahaay.audio import WasapiSource

    print("\n  Devices")
    try:
        devices = WasapiSource.list_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! cannot enumerate devices: {exc}")
        print("    pip install pyaudiowpatch")
        return 1

    if not devices:
        print("  ! no input devices at all")
        print("    pip install pyaudiowpatch")
        return 1

    for d in devices[:12]:
        kind = "loopback" if d.is_loopback else "input   "
        print(f"    [{d.index:>2}] {kind}  {d.name[:58]}")
    loopbacks = [d for d in devices if d.is_loopback]
    print(f"\n    {len(devices)} input devices, {len(loopbacks)} loopback")
    if not loopbacks:
        print("    ! no loopback device - system audio cannot be captured.")
        print("      The microphone may still work.")

    source = WasapiSource(cfg.audio)
    try:
        source.start()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ! capture would not start: {exc}")
        return 1

    tone = None
    if args.play:
        tone = Path(tempfile.gettempdir()) / "sahaay_check_tone.wav"
        write_tone(tone, args.seconds + 1.0)
        print("\n  Playing a test tone through the default output device...")
        play_async(tone)
    else:
        print("\n  Listening. Play something - a video, music, your own voice.")

    print(f"  Listening for {args.seconds:.0f} s\n")

    blocks: list[np.ndarray] = []
    deadline = time.time() + args.seconds
    last_report = 0.0
    started = time.time()

    while time.time() < deadline:
        chunk = source.read(timeout=0.5)
        if chunk is None or not chunk.size:
            continue
        blocks.append(chunk)
        elapsed = time.time() - started
        if elapsed - last_report >= 1.0:
            last_report = elapsed
            recent = np.concatenate(blocks[-40:]) if blocks else np.zeros(1)
            rms = float(np.sqrt(np.mean(np.square(recent))))
            bar = "#" * min(40, int(rms * 400))
            print(f"    {elapsed:4.1f}s  rms {rms:8.5f}  {bar}")

    source.stop()
    if tone is not None:
        try:
            import winsound

            winsound.PlaySound(None, 0)
            tone.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    if not blocks:
        print("\n  FAIL - the capture stream opened and delivered no audio at all.")
        print("  Nothing reached the pipeline, so no caption could ever appear.")
        return 1

    audio = np.concatenate(blocks)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    seconds = audio.size / cfg.audio.sample_rate

    print(f"\n  Captured {seconds:.1f}s   rms {rms:.5f}   peak {peak:.3f}")

    if rms < SILENCE_RMS:
        print("\n  FAIL - the stream is running but it is silence.")
        print("  Common causes, in the order worth checking:")
        print("    - nothing is playing (with --play, the output device is muted)")
        print("    - Windows is outputting to a different device than the one")
        print("      captured; change the default output device and re-run")
        print("    - the app's volume mixer entry is muted")
        return 1

    if rms < QUIET_RMS:
        print("\n  DEGRADED - audio is arriving but it is very quiet.")
        print("  Whisper will transcribe it badly. Raise the system volume")
        print("  and re-run; aim for an rms above 0.002.")
        return 1

    print("\n  OK - real audio is reaching the pipeline.")
    print("  A lecture playing now would caption. Start the app with run.bat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
