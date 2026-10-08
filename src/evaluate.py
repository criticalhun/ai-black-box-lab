"""src/evaluate.py — Accuracy / loss on the answer position."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    logits = model(x)[:, -1, :].to(torch.float64)
    return {"acc": (logits.argmax(dim=-1) == y).double().mean().item(),
            "loss": F.cross_entropy(logits, y).item()}
