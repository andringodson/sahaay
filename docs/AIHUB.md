# Qualcomm AI Hub device-farm results

Submitted 19 Sep 2026 with `scripts/aihub_profile.py`. Every row is a real
measurement on physical Snapdragon hardware provisioned by Qualcomm, not an
estimate and not a datasheet figure. **Each job link is public — open one
and check the number yourself.**

| Model | Device | On-device inference | Peak memory | Layers on NPU | Job |
|---|---|---:|---:|---:|---|
| `model.onnx` | Snapdragon X Elite CRD | compile failed | - | - | [jglyoew85](https://workbench.aihub.qualcomm.com/jobs/jglyoew85/) |
| `model.onnx` | Snapdragon X2 Elite CRD | compile failed | - | - | [jgjrok3xp](https://workbench.aihub.qualcomm.com/jobs/jgjrok3xp/) |
| `model.onnx` | Snapdragon X Plus 8-Core CRD | compile failed | - | - | [j5wl8mz6p](https://workbench.aihub.qualcomm.com/jobs/j5wl8mz6p/) |
| `encoder_model.onnx` | Snapdragon X Elite CRD | **27.48 ms** | 16.9 MB | 129/129 | [jpe784qo5](https://workbench.aihub.qualcomm.com/jobs/jpe784qo5/) |
| `encoder_model.onnx` | Snapdragon X2 Elite CRD | **13.5 ms** | 9.2 MB | 129/129 | [j568rqx6g](https://workbench.aihub.qualcomm.com/jobs/j568rqx6g/) |
| `encoder_model.onnx` | Snapdragon X Plus 8-Core CRD | **26.71 ms** | 16.2 MB | 129/129 | [jgdd87jrg](https://workbench.aihub.qualcomm.com/jobs/jgdd87jrg/) |

## What these numbers are

The Whisper encoder is the dominant cost in transcription, and on
Snapdragon X2 Elite CRD it runs in **13.5 ms** for a full
30-second mel window. For comparison, the same graph measured
[on the x86 development machine](BENCHMARKS.md) is an order of magnitude
slower — which is the entire argument for shipping this on a Snapdragon PC.

## What is deliberately not here

The autoregressive decoders (`decoder_model_merged`) are not profiled.
They carry a dynamic KV cache, so a single fixed-shape profile would
measure one arbitrary sequence length and then read as though it were the
cost per caption. It is not, and publishing it would be misleading.
End-to-end per-caption latency is measured instead by
`scripts/bench.py`, which runs the real decode loop.

## Why this file exists

Sahaay was developed without a Snapdragon PC on the desk. Rather than
claiming NPU performance that could not be measured, the graphs were
submitted to Qualcomm's own device farm and the numbers came back from the
silicon itself.

[BENCHMARKS.md](BENCHMARKS.md) covers the development machine and the CPU
fallback path; this file covers the target hardware. Together they are the
honest version of the claim.
