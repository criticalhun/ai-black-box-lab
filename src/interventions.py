"""src/interventions.py — Weight-level interventions. Always act on a COPY."""
from __future__ import annotations

import copy

import torch

from src.fourier import project_frequencies
from src.fourier2d import project_out_classes
from src.model import Transformer


def embed_keep_frequencies(model: Transformer, keep: set[int],
                           rows: tuple[int, int] = (0, 113)) -> Transformer:
    """Return a copy of `model` whose W_E[rows] keeps only the given Fourier frequencies.
    All other weights (incl. the '=' row) are unchanged."""
    m = copy.deepcopy(model)
    lo, hi = rows
    with torch.no_grad():
        W = m.W_E[lo:hi]
        m.W_E[lo:hi] = project_frequencies(W, keep).to(dtype=W.dtype, device=W.device)
    return m


def mlp_class_ablation_hook(p: int, remove: set[str], pos: int = 2):
    """Forward hook for 'blocks.0.mlp.hook_post'. Requires the batch to be the FULL
    (a, b) grid in a-major order (N = p*p, row a*p+b). Removes the given 2D Fourier
    classes from the activations at position `pos`; other positions untouched."""
    def hook(act: torch.Tensor, name: str) -> torch.Tensor:
        if act.shape[0] != p * p:
            raise ValueError(f"{name}: batch must be the full {p}x{p} grid, got {act.shape[0]}")
        A = act[:, pos, :].reshape(p, p, -1)
        A2 = project_out_classes(A, remove).to(dtype=act.dtype, device=act.device)
        out = act.clone()
        out[:, pos, :] = A2.reshape(p * p, -1)
        return out
    return hook


def mean_ablation_hook():
    """Forward hook: replace an activation by its mean over the batch (separately for
    each position). The component keeps its average effect but carries no
    input-specific information. Use with the full (a, b) grid as the batch."""
    def hook(act: torch.Tensor, name: str) -> torch.Tensor:
        return act.mean(dim=0, keepdim=True).expand_as(act).clone()
    return hook
