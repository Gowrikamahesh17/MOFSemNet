"""Single resolution point for "which device do we run on".

Every module in src/language, src/graphs, src/sparsification_baselines, src/evaluation and
src/explainability that needs a torch.device should call get_device() from here rather than
writing `torch.device("cuda" if torch.cuda.is_available() else "cpu")` itself — that keeps the
device choice config-driven (PipelineConfig.device) and means switching machines (personal GPU,
Colab, university server) never requires touching more than one place.

src/baseline/ is left untouched (ported as-is per the porting instructions) even though its
gnn_training.py resolves its own device inline — this module is only used by the new Steps 2-6
work.
"""

from __future__ import annotations

import torch

from src.common.config import RuntimeConfig

VALID_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def get_device(config: RuntimeConfig | None = None) -> torch.device:
    """Resolve the torch.device to use, based on config.device.

    "auto" (the default) picks CUDA if available, then Apple Silicon MPS, then falls back to
    CPU — so the same code runs unmodified on a personal GPU box, Google Colab, a university
    server, or a laptop. "cpu"/"cuda"/"mps" force that backend explicitly (useful for tests or
    for isolating a device-specific bug); requesting an unavailable backend raises rather than
    silently falling back, since a silent fallback would hide a broken CUDA/MPS setup.
    """
    device_choice = config.device if config is not None else "auto"

    if device_choice not in VALID_DEVICE_CHOICES:
        raise ValueError(f"device must be one of {VALID_DEVICE_CHOICES}, got {device_choice!r}")

    if device_choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but no CUDA device is available on this machine.")
    if device_choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available on this machine.")

    return torch.device(device_choice)
