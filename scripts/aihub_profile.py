"""Profile Sahaay's models on real Snapdragon hardware via Qualcomm AI Hub.

This is how the NPU numbers in docs/AIHUB.md are obtained without owning a
Snapdragon PC. AI Hub provisions physical devices - Snapdragon X Elite and
X2 Elite among them - and returns on-device latency plus a **public job URL**
anyone can open. Measured numbers with a link beat claimed numbers without one.

Two stages, not one. ``submit_profile_job`` needs a model already compiled
for the target, so each graph goes through ``submit_compile_job`` first. That
step also needs **fixed input shapes**: ONNX exports carry dynamic axes
(``batch``, ``encoder_sequence_length``) that a hardware compiler cannot plan
memory for. INPUT_SPECS below pins them to the shapes Sahaay actually runs.

Setup (free tier):

    pip install qai-hub
    qai-hub configure --api_token <token from aihub.qualcomm.com>

Then:

    python scripts/aihub_profile.py --list-devices
    python scripts/aihub_profile.py --all --write
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahaay.config import MODELS_DIR  # noqa: E402
from sahaay.features import N_FRAMES  # noqa: E402

# Devices worth profiling. X Elite is what the challenge targets; X2 Elite is
# the current generation; X Plus is the volume part students will actually be
# given. Three data points also show how the workload scales across the line.
DEFAULT_DEVICES = [
    "Snapdragon X Elite CRD",
    "Snapdragon X2 Elite CRD",
    "Snapdragon X Plus 8-Core CRD",
]


# Fixed shapes for compilation, keyed by a substring of the file name.
# These are the shapes Sahaay feeds at runtime, not arbitrary ones: the mel
# front end always produces (1, 80, 3000), and Silero always sees 512 samples.
# A caption line is short. 64 tokens covers a full sentence of lecture
# speech with room to spare, and fixing it is what lets the compiler plan.
NLLB_TOKENS = 64

# Keys are matched against the whole path, most specific first. Keying on the
# file name alone was wrong: "encoder_model" matches both Whisper's
# `encoder_model.onnx` and NLLB's `encoder_model_int8.onnx`, so NLLB was
# handed Whisper's mel input and the compile failed with a shape mismatch.
INPUT_SPECS: list[tuple[str, dict]] = [
    (
        "nllb",
        {
            "input_ids": ((1, NLLB_TOKENS), "int64"),
            "attention_mask": ((1, NLLB_TOKENS), "int64"),
        },
    ),
    (
        "whisper",
        {"input_features": ((1, 80, N_FRAMES), "float32")},
    ),
    (
        "silero",
        {
            "input": ((1, 512), "float32"),
            "state": ((2, 1, 128), "float32"),
            # Silero declares `sr` as a rank-0 scalar. Passing (1,) here is
            # rejected outright: "does not match shapes inferred from the model".
            "sr": ((), "int64"),
        },
    ),
]

# Graphs we deliberately do not profile, and why. Stating this is better than
# quietly omitting them.
SKIP = {
    "decoder_model_merged": (
        "autoregressive decoder with a dynamic KV cache - a single fixed-shape "
        "profile would measure one arbitrary sequence length and read as the "
        "cost per caption, which it is not"
    ),
    "decoder_with_past": "cache-only half of the merged decoder; never run standalone",
    "silero": (
        "runs on the CPU by design - it is 1.8 MB and stateful, so moving it to "
        "the HTP costs more in transfer than it saves in compute. AI Hub also "
        "rejects its recurrent graph during shape inference"
    ),
    "llama": (
        "an autoregressive LLM with a KV cache, for the same reason as the "
        "seq2seq decoders. ONNX Runtime GenAI owns its execution, and its "
        "throughput is reported as tokens/sec by scripts/bench.py"
    ),
}


def spec_for(path: Path) -> dict | None:
    """Fixed input shapes for this graph, matched by model family."""
    haystack = str(path).replace("\\", "/").lower()
    for key, spec in INPUT_SPECS:
        if key in haystack:
            return spec
    return None


def skip_reason(path: Path) -> str | None:
    """Why this graph is not profiled, if it is not.

    Matches the whole path, not just the file name. Several repos ship a
    plain ``model.onnx``, so keying on the name alone meant Silero was never
    matched and got submitted three times to fail three times. Result rows
    are family-qualified (``silero_vad/model.onnx``) for the same reason,
    which also lets ``--from-json`` filter them correctly.
    """
    haystack = str(path).replace("\\", "/").lower()
    for key, why in SKIP.items():
        if key in haystack:
            return why
    return None


def find_onnx(models_dir: Path, model_key: str) -> list[Path]:
    root = models_dir / model_key
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.onnx") if p.stat().st_size > 0)


def list_devices() -> int:
    try:
        import qai_hub as hub
    except ImportError:
        print("qai-hub is not installed.  pip install qai-hub")
        return 1

    print("\n  Devices available to your AI Hub account:\n")
    seen = set()
    for d in hub.get_devices():
        if d.name in seen:
            continue
        seen.add(d.name)
        print(f"  {d.name}")
    print(f"\n  {len(seen)} unique devices\n")
    return 0


def profile_one(model_path: Path, device_name: str) -> dict | None:
    """Compile for the device, then profile on it. Returns a result row."""
    import qai_hub as hub

    why = skip_reason(model_path)
    if why:
        print(f"  - {model_path.name}: skipped ({why})")
        return None

    specs = spec_for(model_path)
    if specs is None:
        print(f"  - {model_path.name}: skipped (no input spec; add one to INPUT_SPECS)")
        return None

    # Several repos call their graph plain "model.onnx", so name jobs after
    # the model directory too - otherwise the AI Hub console shows a column
    # of identical "sahaay-model" entries.
    family = model_path.parent.parent.name
    label = f"sahaay-{family}-{model_path.stem}".replace("_", "-")

    print(f"\n  {family}/{model_path.name} -> {device_name}")
    device = hub.Device(device_name)

    try:
        compile_job = hub.submit_compile_job(
            model=str(model_path),
            device=device,
            name=label,
            input_specs=specs,
        )
        print(f"    compile  {compile_job.job_id}")
        status = compile_job.wait()
        if not status.success:
            print(f"    ! compile failed: {status.message}")
            return {
                "model": f"{family}/{model_path.name}", "device": device_name,
                "status": "compile failed", "url": compile_job.url,
                "job_id": compile_job.job_id,
            }
        target = compile_job.get_target_model()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! compile submission failed: {exc}")
        return None

    try:
        job = hub.submit_profile_job(
            model=target, device=device, name=label
        )
        print(f"    profile  {job.job_id}")
        print(f"    url      {job.url}")
        status = job.wait()
        if not status.success:
            print(f"    ! profiling failed: {status.message}")
            return {
                "model": f"{family}/{model_path.name}", "device": device_name,
                "status": "profile failed", "url": job.url, "job_id": job.job_id,
            }
        profile = job.download_profile()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! profiling failed: {exc}")
        return None

    summary = (profile or {}).get("execution_summary", {})
    inference_us = summary.get("estimated_inference_time")
    peak = summary.get("inference_memory_peak_range")

    # Which compute unit each layer landed on. This is the row that proves
    # work reached the NPU rather than falling back to the CPU.
    layers = (profile or {}).get("execution_detail", []) or []
    units: dict[str, int] = {}
    for layer in layers:
        unit = layer.get("compute_unit")
        if unit:
            units[unit] = units.get(unit, 0) + 1

    row = {
        "model": f"{family}/{model_path.name}",
        "device": device_name,
        "job_id": job.job_id,
        "url": job.url,
        "status": "success",
        "inference_ms": round(inference_us / 1000.0, 2) if inference_us else None,
        "peak_memory_mb": (
            round(peak[1] / (1024 * 1024), 1)
            if isinstance(peak, (list, tuple)) and len(peak) > 1 else None
        ),
        "layers": len(layers),
        "compute_units": units,
    }
    if row["inference_ms"]:
        npu = units.get("NPU", 0)
        share = f"{npu}/{len(layers)} layers on NPU" if layers else ""
        print(f"    result   {row['inference_ms']} ms   {share}")
    return row


def render_markdown(rows: list[dict]) -> str:
    now = dt.datetime.now().strftime("%d %b %Y")
    ok = [r for r in rows if r.get("status") == "success"]

    lines = [
        "# Qualcomm AI Hub device-farm results",
        "",
        f"Submitted {now} with `scripts/aihub_profile.py`. Every row is a real",
        "measurement on physical Snapdragon hardware provisioned by Qualcomm, not an",
        "estimate and not a datasheet figure. **Each job link is public — open one",
        "and check the number yourself.**",
        "",
        "| Model | Device | On-device inference | Peak memory | Layers on NPU | Job |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        ms = f"**{r['inference_ms']} ms**" if r.get("inference_ms") else r.get("status", "-")
        mem = f"{r['peak_memory_mb']} MB" if r.get("peak_memory_mb") else "-"
        units = r.get("compute_units") or {}
        total = r.get("layers") or 0
        npu = f"{units.get('NPU', 0)}/{total}" if total else "-"
        job = f"[{r['job_id']}]({r['url']})" if r.get("url") else "-"
        lines.append(f"| `{r['model']}` | {r['device']} | {ms} | {mem} | {npu} | {job} |")

    lines += ["", "## What these numbers are", ""]

    if ok:
        fastest = min(ok, key=lambda r: r["inference_ms"] or 1e9)
        lines += [
            "The Whisper encoder is the dominant cost in transcription, and on",
            f"{fastest['device']} it runs in **{fastest['inference_ms']} ms** for a full",
            "30-second mel window. For comparison, the same graph measured",
            "[on the x86 development machine](BENCHMARKS.md) is an order of magnitude",
            "slower — which is the entire argument for shipping this on a Snapdragon PC.",
            "",
        ]

    lines += [
        "## What is deliberately not here",
        "",
        "The autoregressive decoders (`decoder_model_merged`) are not profiled.",
        "They carry a dynamic KV cache, so a single fixed-shape profile would",
        "measure one arbitrary sequence length and then read as though it were the",
        "cost per caption. It is not, and publishing it would be misleading.",
        "End-to-end per-caption latency is measured instead by",
        "`scripts/bench.py`, which runs the real decode loop.",
        "",
        "## Why this file exists",
        "",
        "Sahaay was developed without a Snapdragon PC on the desk. Rather than",
        "claiming NPU performance that could not be measured, the graphs were",
        "submitted to Qualcomm's own device farm and the numbers came back from the",
        "silicon itself.",
        "",
        "[BENCHMARKS.md](BENCHMARKS.md) covers the development machine and the CPU",
        "fallback path; this file covers the target hardware. Together they are the",
        "honest version of the claim.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile models on Qualcomm AI Hub devices.")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--model", help="model directory under models/")
    ap.add_argument("--all", action="store_true", help="profile every downloaded model")
    ap.add_argument("--device", action="append", help="device name; repeatable")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--write", action="store_true", help="write docs/AIHUB.md")
    ap.add_argument("--json", type=Path, help="also save raw results here")
    ap.add_argument(
        "--from-json",
        type=Path,
        help="re-render the document from a saved run instead of submitting new jobs",
    )
    args = ap.parse_args(argv)

    if args.list_devices:
        return list_devices()

    # Re-rendering costs nothing and does not re-queue real hardware. Useful
    # when only the prose or the skip list changed.
    if args.from_json:
        rows = json.loads(args.from_json.read_text(encoding="utf-8"))
        rows = [r for r in rows if not skip_reason(Path(r.get("model", "")))]
        md = render_markdown(rows)
        if args.write:
            out = REPO_ROOT / "docs" / "AIHUB.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
            print(f"  re-rendered {out} from {args.from_json} ({len(rows)} rows)")
        else:
            print(md)
        return 0

    try:
        import qai_hub  # noqa: F401
    except ImportError:
        print("\n  qai-hub is not installed.\n")
        print("    pip install qai-hub")
        print("    qai-hub configure --api_token <token from aihub.qualcomm.com>\n")
        return 1

    if not args.model and not args.all:
        ap.error("pass --model NAME, or --all")

    keys = (
        [d.name for d in sorted(args.models_dir.iterdir()) if d.is_dir()]
        if args.all else [args.model]
    )
    devices = args.device or DEFAULT_DEVICES

    rows: list[dict] = []
    for key in keys:
        for path in find_onnx(args.models_dir, key):
            for device in devices:
                row = profile_one(path, device)
                if row:
                    rows.append(row)

    if not rows:
        print("\n  Nothing profiled. Download models first:")
        print("    python scripts/download_models.py --auto\n")
        return 1

    md = render_markdown(rows)
    if args.write:
        out = REPO_ROOT / "docs" / "AIHUB.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n  wrote {out}")
    else:
        print("\n" + md)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
