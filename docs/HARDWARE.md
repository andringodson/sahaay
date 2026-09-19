# Hardware notes

## The trap: `pip install onnxruntime-qnn` is not enough

This cost more time than anything else in the build, and it fails **silently**, so it is worth writing down properly.

Since ONNX Runtime 1.23, Qualcomm's execution provider is a **plugin** EP. `pip install onnxruntime-qnn` lays down a *separate* package — `onnxruntime_qnn` — containing `onnxruntime_providers_qnn.dll` and the QNN backend libraries. ONNX Runtime does not load it on its own.

Measured on the development machine:

```python
>>> import onnxruntime as ort
>>> ort.get_available_providers()
['AzureExecutionProvider', 'CPUExecutionProvider']          # no QNN

>>> import onnxruntime_qnn as oq
>>> ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
>>> ort.get_available_providers()
['AzureExecutionProvider', 'CPUExecutionProvider', 'QNNExecutionProvider']
```

So the obvious implementation —

```python
providers = ["QNNExecutionProvider", "CPUExecutionProvider"]   # WRONG
session = ort.InferenceSession(model, providers=providers)
```

— quietly runs on the CPU **on a Snapdragon device**, with everything appearing to work. The app would have shipped claiming NPU acceleration it never used.

`sahaay/runtime.py` performs the registration on import. `tests/test_runtime.py::TestRegistration` asserts it happened.

## The second trap: registered ≠ NPU present

Registering the plugin succeeds on machines with no Hexagon hardware at all. On an x86 development box:

```python
>>> for d in ort.get_ep_devices():
...     print(d.ep_name, d.device.type)
CPUExecutionProvider   OrtHardwareDeviceType.CPU
QNNExecutionProvider   OrtHardwareDeviceType.CPU      # <- registered, but CPU
```

The QNN provider is available and would be selected by a naive priority walk — then execute on the CPU, more slowly than the plain CPU provider, while the UI reported "Hexagon NPU".

So `npu_active` checks `get_ep_devices()` for an actual `NPU` hardware type, and the selector **skips** a CPU-bound QNN provider rather than using it. The fallback reason distinguishes the two cases, because they look identical in the provider list and mean completely different things:

- `Dml not installed` — the provider isn't there
- `QNN present but bound to CPU, so there is no Hexagon NPU here` — the provider is there, the silicon isn't

## The third trap: `backend_path`

Most documentation shows:

```python
provider_options = {"backend_path": "QnnHtp.dll"}
```

That relies on `QnnHtp.dll` being on `PATH`. It is not — it lives inside the wheel, at `site-packages/onnxruntime_qnn/libs/<arch>/`. The package resolves the correct `amd64` / `arm64ec` / `arm64` subdirectory, so Sahaay passes the full path from `onnxruntime_qnn.get_qnn_htp_path()`.

## Requirements

| | |
|---|---|
| Python | **3.11+** for `onnxruntime-qnn` wheels. On 3.10 or older you get stock onnxruntime and no NPU path at all. |
| OS | Windows 11. WASAPI loopback capture is Windows-specific. |
| ONNX Runtime | 1.23+ for plugin EPs; developed against 1.30.0 with `onnxruntime-qnn` 2.6.0. |
| Target | Snapdragon X Elite / X Plus / X2 (Copilot+ class). |

Quantisation utilities in `onnxruntime` are x86-64 only. This does not affect Sahaay — every model is downloaded pre-quantised — but it matters if you want to quantise your own.

## Developing without a Snapdragon PC

This project was built on an x86 machine. Three things made that workable, and none of them involve pretending:

1. **The QNN wheel installs on x64.** The registration path, the provider options and the fallback logic are all exercised locally — just bound to CPU rather than HTP.
2. **`SAHAAY_PROVIDER` forces a provider**, so the CPU column of the benchmark table is reproducible anywhere.
3. **The Qualcomm AI Hub device farm provides real hardware, free.** `scripts/aihub_profile.py` submits the graphs to a physical Snapdragon X Elite and returns latency plus a public job URL. Those are genuine on-device measurements that a reviewer can independently open.

What is *not* claimed: end-to-end wall-clock numbers of the full three-model pipeline on an X Elite. Those need the device. `docs/BENCHMARKS.md` states which machine produced every row.

## Verifying on a real device

```powershell
.\run.bat --device                                    # expect "Hexagon NPU active : yes"
python scripts\bench.py --compare --write             # NPU vs CPU, same machine
python scripts\aihub_profile.py --all --write         # device-farm cross-check
```

If `--device` says `no` on a Snapdragon PC, check in this order:

1. Python is 3.11+ (`.\run.bat --device` prints nothing about QNN on 3.10)
2. `onnxruntime-qnn` installed, not just `onnxruntime`
3. `pip show onnxruntime-qnn` reports 2.5.0 or newer
4. Run with `-v` — the registration attempt and its failure reason are logged
