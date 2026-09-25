/* Capture on the audio thread, so inference cannot starve it.
 *
 * Whisper runs as WebAssembly on the main thread, and a single transcription
 * blocks it for seconds. A ScriptProcessorNode delivers its callbacks on that
 * same thread, so while a segment was being transcribed the next one was not
 * being recorded - measured as whole sentences missing from the transcript.
 *
 * An AudioWorkletProcessor runs on the audio rendering thread instead. It
 * keeps filling buffers through a blocked main thread and posts them across;
 * the messages queue and are delivered when the main thread comes back. The
 * captions arrive late under load, which is honest, rather than the audio
 * disappearing, which is not.
 *
 * The render quantum is 128 frames; this accumulates to 512, which is the
 * 32 ms frame the desktop pipeline segments on.
 */

const FRAME = 512;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(FRAME);
    this._at = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this._buffer[this._at++] = channel[i];
      if (this._at === FRAME) {
        // slice() copies: the buffer is reused on the next quantum, and
        // posting a view of it would deliver whatever it held by then.
        this.port.postMessage(this._buffer.slice());
        this._at = 0;
      }
    }
    return true;
  }
}

registerProcessor("sahaay-capture", CaptureProcessor);
