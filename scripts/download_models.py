"""Fetch model weights into models/.

Nothing is vendored into the repo: the weights are hundreds of megabytes and
several are licence-gated. This script is the reproducible substitute.

Two tiers per model. The ``snapdragon`` tier is the pre-quantised,
pre-compiled asset that targets the Hexagon NPU; the ``portable`` tier is a
generic ONNX export that runs anywhere. ``--auto`` picks by architecture, so
the same command does the right thing on an HP Omnibook and on an x86
laptop.

    python scripts/download_models.py --auto          # recommended
    python scripts/download_models.py --all           # both tiers
    python scripts/download_models.py --vad --asr     # just these
    python scripts/download_models.py --list          # show sizes, download nothing
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahaay.config import MODELS_DIR  # noqa: E402


@dataclass
class ModelSpec:
    key: str
    target_dir: str
    repo_id: str
    description: str
    approx_mb: int
    allow_patterns: list[str] = field(default_factory=list)
    tier: str = "portable"
    note: str = ""


# Anything larger than ~2 GB is called out explicitly, because a student on a
# metered connection deserves to know before the download starts.
MODELS: list[ModelSpec] = [
    ModelSpec(
        key="vad",
        target_dir="silero_vad",
        repo_id="onnx-community/silero-vad",
        description="Silero VAD - speech/silence gating",
        approx_mb=2,
        allow_patterns=["*.onnx", "*.json"],
        tier="both",
        note="Tiny; always runs on CPU. Without it Sahaay falls back to energy gating.",
    ),
    ModelSpec(
        key="asr",
        target_dir="whisper_small_quantized",
        repo_id="qualcomm/Whisper-Small-Quantized",
        description="Whisper Small, w8a16, precompiled QNN ONNX",
        approx_mb=520,
        allow_patterns=["*.onnx", "*.onnx.data", "*.json", "*.bin", "*.txt"],
        tier="snapdragon",
        note="Validated by Qualcomm on Snapdragon X Elite. This is the fast path.",
    ),
    ModelSpec(
        key="asr",
        target_dir="whisper_small_portable",
        repo_id="onnx-community/whisper-small",
        description="Whisper Small, generic ONNX export",
        approx_mb=980,
        allow_patterns=["onnx/*.onnx", "onnx/*.onnx_data", "*.json"],
        tier="portable",
        note="Runs on any CPU. Slower, but it is what makes the repo reviewable.",
    ),
    ModelSpec(
        key="translate",
        target_dir="nllb_200_distilled_600m_int8",
        repo_id="Xenova/nllb-200-distilled-600M",
        description="NLLB-200 distilled 600M, INT8 - 200 languages",
        approx_mb=650,
        allow_patterns=["onnx/*quantized*.onnx", "onnx/*int8*.onnx", "*.json", "*.model"],
        tier="both",
        note="Covers all eight Indian target languages in one model.",
    ),
    ModelSpec(
        key="llm",
        target_dir="llama_3_2_3b_instruct_hexagon",
        repo_id="onnx-community/Llama-3.2-3B-instruct-hexagon-npu-assets",
        description="Llama 3.2 3B Instruct, prebuilt Hexagon NPU assets",
        approx_mb=2600,
        tier="snapdragon",
        note="Prebuilt QNN context binaries - skips the local AI Hub compile entirely.",
    ),
    ModelSpec(
        key="llm",
        target_dir="llama_3_2_3b_instruct_portable",
        repo_id="onnx-community/Llama-3.2-3B-Instruct-ONNX",
        description="Llama 3.2 3B Instruct, generic ONNX (int4)",
        approx_mb=2100,
        allow_patterns=["onnx/*int4*", "*.json", "*.model"],
        tier="portable",
        note="Optional. Without any LLM the glossary uses the seeded heuristic.",
    ),
]


def detect_tier() -> str:
    """ARM64 Windows means a Snapdragon PC in every realistic case here."""
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"} and platform.system() == "Windows":
        return "snapdragon"
    return "portable"


def wanted(spec: ModelSpec, tier: str) -> bool:
    return spec.tier in (tier, "both")


def human(mb: int) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


def show_plan(specs: list[ModelSpec], tier: str) -> None:
    total = sum(s.approx_mb for s in specs)
    print(f"\n  Tier: {tier}   ({len(specs)} downloads, about {human(total)})\n")
    for s in specs:
        print(f"  {s.target_dir:<34} {human(s.approx_mb):>8}   {s.description}")
        if s.note:
            print(f"  {'':<34} {'':>8}   {s.note}")
    print()


def download(spec: ModelSpec, models_dir: Path, force: bool = False) -> bool:
    from huggingface_hub import snapshot_download

    dest = models_dir / spec.target_dir
    if dest.exists() and any(dest.rglob("*.onnx")) and not force:
        print(f"  = {spec.target_dir} (already present)")
        return True

    print(f"  > {spec.target_dir}  <- {spec.repo_id}  ({human(spec.approx_mb)})")
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(dest),
            allow_patterns=spec.allow_patterns or None,
            # Symlinks into the HF cache break when the cache is cleared, and
            # the whole point here is a self-contained models/ directory.
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! failed: {exc}")
        if "gated" in str(exc).lower() or "401" in str(exc):
            print("    This repo is licence-gated. Accept the terms on the model page,")
            print("    then run:  huggingface-cli login")
        return False
    print(f"  + {spec.target_dir} ready")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download Sahaay model weights.")
    ap.add_argument("--auto", action="store_true", help="pick the tier for this machine")
    ap.add_argument("--all", action="store_true", help="download every tier")
    ap.add_argument("--tier", choices=["snapdragon", "portable"], help="force a tier")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--list", action="store_true", help="show the plan and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)

    keys = {k for k in ("vad", "asr", "translate", "llm") if getattr(args, k)}
    tier = args.tier or ("portable" if args.all else detect_tier())

    specs = [s for s in MODELS if (not keys or s.key in keys)]
    if not args.all:
        specs = [s for s in specs if wanted(s, tier)]

    if not specs:
        print("Nothing selected. Try --auto, or --list to see the options.")
        return 1

    show_plan(specs, "all" if args.all else tier)
    if args.list:
        return 0

    args.models_dir.mkdir(parents=True, exist_ok=True)
    ok = sum(download(s, args.models_dir, force=args.force) for s in specs)

    print(f"\n  {ok}/{len(specs)} ready in {args.models_dir}")
    if ok < len(specs):
        print("  Sahaay still runs with what downloaded - missing stages degrade.")
    print("  Check what will execute:  python -m sahaay --device\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
