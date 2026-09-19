"""Sahaay - offline live lecture captioning for Snapdragon-powered HP PCs.

Everything in this package is designed to run with no network connection:
audio never leaves the device, and every model executes on the local
Hexagon NPU (or a CPU/DirectML fallback when the NPU is not present).
"""

__version__ = "0.3.0"
__all__ = ["__version__"]
