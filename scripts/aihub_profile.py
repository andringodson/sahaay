"""Profile Sahaay's models on real Snapdragon hardware via Qualcomm AI Hub.

This is how the NPU numbers in docs/BENCHMARKS.md are obtained without owning
a Snapdragon PC. AI Hub provisions physical devices - Snapdragon X Elite
among them - and returns per-layer latency plus a **public job URL** that
anyone can open and verify. Measured numbers with a link beat claimed numbers
without one.

Setup (free tier):

    pip install qai-hub
    qai-hub configure --api_token <token from aihub.qualcomm.com>

Then:

    python scripts/aihub_profile.py --list-devices
    python scripts/aihub_profile.py --model whisper_small_quantized
    python scripts/aihub_profile.py --all --write

The job links it prints go straight into the README. A reviewer clicking one
sees Qualcomm's own measurement of our graph on their own silicon, which is a
stronger claim than anything this repo could assert about itself.
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

# Devices worth profiling against. The X Elite is the one the challenge
# targets; the 8 Gen 3 is included because the same graphs are what a phone
# build would use, and it costs nothing extra to ask.
DEFAULT_DEVICES = [
    "Snapdragon X Elite CRD",
    "Snapdragon 8 Elite QRD",
]


def find_onnx(models_dir: Path, model_key: str) -> list[Path]:
    root = models_dir / model_key
    if not root.exists():
        return []
    # Skip the huge external-data blobs; AI Hub takes the graph file.
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
        attrs = ", ".join(sorted(a for a in d.attributes if a.startswith("chipset")))
        print(f"  {d.name:<36} {attrs}")
    print()
    return 0


def profile(model_path: Path, device_name: str, verbose: bool = True) -> dict | None:
    import qai_hub as hub

    if verbose:
        print(f"\n  submitting {model_path.name} -> {device_name}")
    try:
        device = hub.Device(device_name)
        job = hub.submit_profile_job(
            model=str(model_path),
            device=device,
            name=f"sahaay-{model_path.stem}",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! submission failed: {exc}")
        return None

    url = getattr(job, "url", None) or f"https://aihub.qualcomm.com/jobs/{job.job_id}"
    print(f"    job    {job.job_id}")
    print(f"    url    {url}")
    print("    waiting for the device to report...")

    try:
        result = job.download_profile()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! profiling did not complete: {exc}")
        print(f"    the job may still finish - check {url}")
        return {"model": model_path.name, "device": device_name, "job_id": job.job_id,
                "url": url, "status": "pending"}

    summary = (result or {}).get("execution_summary", {})
    inference_us = summary.get("estimated_inference_time")
    peak_memory = summary.get("inference_memory_peak_range")

    row = {
        "model": model_path.name,
        "device": device_name,
        "job_id": job.job_id,
        "url": url,
        "status": "success",
        "inference_ms": round(inference_us / 1000.0, 2) if inference_us else None,
        "peak_memory_mb": (
            round(peak_memory[1] / (1024 * 1024), 1)
            if isinstance(peak_memory, (list, tuple)) and len(peak_memory) > 1 else None
        ),
        "compute_units": summary.get("compute_unit_breakdown"),
    }
    if row["inference_ms"]:
        print(f"    result {row['inference_ms']} ms on device")
    return row


def render_markdown(rows: list[dict]) -> str:
    now = dt.datetime.now().strftime("%d %b %Y")
    lines = [
        "# Qualcomm AI Hub device-farm results",
        "",
        f"Submitted {now} with `scripts/aihub_profile.py`. Each row is a real",
        "measurement on physical Snapdragon hardware provisioned by Qualcomm, not an",
        "estimate. Every job link is public - open one to verify the number.",
        "",
        "| Model | Device | On-device inference | Peak memory | Job |",
        "|---|---|---:|---:|---|",
    ]
    for r in rows:
        ms = f"{r['inference_ms']} ms" if r.get("inference_ms") else r.get("status", "-")
        mem = f"{r['peak_memory_mb']} MB" if r.get("peak_memory_mb") else "-"
        lines.append(
            f"| `{r['model']}` | {r['device']} | {ms} | {mem} | [{r['job_id']}]({r['url']}) |"
        )

    lines += [
        "",
        "## Why this table exists",
        "",
        "Sahaay was developed without a Snapdragon PC on the desk. Rather than",
        "claiming NPU performance we could not measure, the graphs were submitted to",
        "the AI Hub device farm and the numbers came back from the silicon itself.",
        "",
        "The local figures in [BENCHMARKS.md](BENCHMARKS.md) cover the development",
        "machine and the CPU fallback path; this table covers the target hardware.",
        "Together they are the honest version of the claim.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile models on Qualcomm AI Hub devices.")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--model", help="model directory under models/ (e.g. whisper_small_quantized)")
    ap.add_argument("--all", action="store_true", help="profile every downloaded model")
    ap.add_argument("--device", action="append", help="device name; repeatable")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--write", action="store_true", help="write docs/AIHUB.md")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    if args.list_devices:
        return list_devices()

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
        paths = find_onnx(args.models_dir, key)
        if not paths:
            print(f"  - {key}: no .onnx found, skipping")
            continue
        for path in paths:
            for device in devices:
                row = profile(path, device)
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
