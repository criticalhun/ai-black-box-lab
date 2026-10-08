"""src/interventions.py — Weight-level interventions. Always act on a COPY."""
from __future__ import annotations

import copy

import torch

from src.fourier import project_frequencies
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
