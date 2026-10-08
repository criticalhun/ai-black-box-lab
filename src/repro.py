"""src/repro.py — Reproducibility helpers."""
from __future__ import annotations

import hashlib
import os

import torch


def setup_determinism(cublas_workspace_config: str = ":4096:8",
                      float32_matmul_precision: str = "highest") -> None:
    """Call BEFORE any CUDA work. Fails loudly if it is too late."""
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current != cublas_workspace_config:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            raise RuntimeError("CUDA already initialized; CUBLAS_WORKSPACE_CONFIG must be set earlier")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision(float32_matmul_precision)  # "highest" = no TF32


def state_sha256(model: torch.nn.Module) -> str:
    """Bit-exact fingerprint of all persistent weights (device-independent format)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
