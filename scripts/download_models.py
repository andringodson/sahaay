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
    # NOTE ON PATTERNS. These repos ship a full-precision graph plus seven or
    # more quantised siblings (int8, fp16, q4, bnb4, ...). A pattern as loose
    # as "onnx/*.onnx" pulls every one of them - measured at multiple GB for
    # whisper-small against an advertised 980 MB. Name the files explicitly.
    ModelSpec(
        key="vad",
        target_dir="silero_vad",
        repo_id="onnx-community/silero-vad",
        description="Silero VAD - speech/silence gating",
        approx_mb=2,
        allow_patterns=["onnx/model.onnx", "*.json"],
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
        description="Whisper Small, generic ONNX export (int8 decoder)",
        approx_mb=620,
        allow_patterns=[
            "onnx/encoder_model.onnx",
            "onnx/decoder_model_merged_int8.onnx",
            "*.json",
        ],
        tier="portable",
        note="Runs on any CPU. Slower, but it is what makes the repo reviewable.",
    ),
    ModelSpec(
        key="asr",
        target_dir="whisper_tiny_en",
        repo_id="onnx-community/whisper-tiny.en",
        description="Whisper Tiny (English) - fast smoke-test model",
        approx_mb=150,
        allow_patterns=[
            "onnx/encoder_model.onnx",
            "onnx/decoder_model_merged.onnx",
            "*.json",
        ],
        tier="portable",
        note="Not for real use - small enough to verify the pipeline in seconds.",
    ),
    ModelSpec(
        key="translate",
        target_dir="nllb_200_distilled_600m_int8",
        repo_id="Xenova/nllb-200-distilled-600M",
        description="NLLB-200 distilled 600M, INT8 - 200 languages",
        approx_mb=650,
        allow_patterns=[
            "onnx/encoder_model_int8.onnx",
            "onnx/decoder_model_merged_int8.onnx",
            "*.json",
            "*.model",
        ],
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
    # The GENAI-ONNX builds are the ones ONNX Runtime GenAI can load directly:
    # they ship genai_config.json alongside the graph. A plain ONNX export
    # does not, and GenAI will not load it.
    ModelSpec(
        key="llm",
        target_dir="llama_3_2_3b_instruct_genai",
        repo_id="onnx-community/Llama-3.2-3B-Instruct-GENAI-ONNX",
        description="Llama 3.2 3B Instruct, GenAI ONNX (int4, CPU/mobile)",
        approx_mb=3500,
        allow_patterns=["cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*"],
        tier="portable",
        note="The glossary model the product ships. Large; skip it to use the 1B.",
    ),
    ModelSpec(
        key="llm",
        target_dir="llama_3_2_1b_instruct_genai",
        repo_id="onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX",
        description="Llama 3.2 1B Instruct, GenAI ONNX (int4, CPU/mobile)",
        approx_mb=1780,
        allow_patterns=["cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*"],
        tier="portable",
        note="Half the download and adequate for glossary lookups. Set glossary.model_id to use it.",
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


def npu_active() -> bool:
    """Will this machine actually execute on the Hexagon NPU?

    Not the same question as the tier. The tier asks "is this ARM64 Windows";
    this asks onnxruntime what the QNN provider is bound to, which is what
    decides which glossary model the app loads.
    """
    try:
        from sahaay.runtime import SessionFactory

        return bool(SessionFactory().npu_active)
    except Exception:  # noqa: BLE001
        # No onnxruntime yet, or no QNN. Either way: not the NPU.
        return False


def prefer_one_llm(specs: list[ModelSpec]) -> tuple[list[ModelSpec], str | None]:
    """Download the glossary model this machine will load, not both.

    The portable tier carries a 3B and a 1B, and `--auto` used to fetch both:
    5.1 GB of a 6.5 GB download for two models where the app only ever loads
    one. Which one is not a preference - `sahaay.llm.order_candidates` picks
    the 3B when the NPU is active and the 1B on CPU, because a glossary entry
    that arrives after the lecture has ended is not a glossary entry.

    `--all` still gets both.
    """
    llms = [s for s in specs if s.key == "llm"]
    if len(llms) < 2:
        return specs, None

    on_npu = npu_active()
    keep = "llama_3_2_3b_instruct_genai" if on_npu else "llama_3_2_1b_instruct_genai"
    dropped = [s for s in llms if s.target_dir != keep]
    if not dropped:
        return specs, None

    saved = human(sum(s.approx_mb for s in dropped))
    why = (
        f"Hexagon NPU active, so the 3B is what runs; skipping the 1B ({saved})."
        if on_npu
        else f"No Hexagon NPU here, so the app loads the 1B on CPU; skipping the 3B ({saved})."
    )
    return [s for s in specs if s not in dropped], why + "  Use --all for both."


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
    note = None
    if not args.all:
        specs = [s for s in specs if wanted(s, tier)]
        specs, note = prefer_one_llm(specs)

    if not specs:
        print("Nothing selected. Try --auto, or --list to see the options.")
        return 1

    show_plan(specs, "all" if args.all else tier)
    if note:
        print(f"  {note}")
        print()
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
