"""src/fourier2d.py — 2D Fourier analysis of functions f(a, b) on Z_p x Z_p."""
from __future__ import annotations

import torch

from src.fourier import basis_frequencies, fourier_basis


def fourier2d_energy(M: torch.Tensor) -> torch.Tensor:
    """M: [p, p, *rest] = f(a, b) (rest e.g. neurons).
    Returns energy grouped by frequency pair (k_a, k_b): [K, K, *rest], K = (p-1)//2 + 1.
    Parseval: result.sum((0, 1)) == (M**2).sum((0, 1))."""
    p = M.shape[0]
    F, _ = fourier_basis(p)
    M = M.detach().cpu().to(torch.float64)
    C = torch.einsum("ia,jb,ab...->ij...", F, F, M)
    e = C ** 2
    bf = basis_frequencies(p)
    K = (p - 1) // 2 + 1
    rows = torch.zeros((K,) + e.shape[1:], dtype=torch.float64).index_add_(0, bf, e)
    return torch.zeros((K, K) + e.shape[2:], dtype=torch.float64).index_add_(1, bf, rows)


def energy_classes(E2: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split [K, K, *rest] energy into: const (0,0), a-only (k,0), b-only (0,k),
    same-frequency products (k,k), cross products (k1,k2 with k1!=k2, both >0)."""
    inner = E2[1:, 1:]
    diag = torch.diagonal(inner, dim1=0, dim2=1).sum(-1)
    return {"const": E2[0, 0], "a_only": E2[1:, 0].sum(0), "b_only": E2[0, 1:].sum(0),
            "same_freq": diag, "cross": inner.sum((0, 1)) - diag,
            "total": E2.sum((0, 1))}


def per_frequency_score(E2: torch.Tensor) -> torch.Tensor:
    """For each k >= 1: E(k,0) + E(0,k) + E(k,k). Shape [K-1, *rest]; row i = frequency i+1."""
    inner = E2[1:, 1:]
    diag = torch.diagonal(inner, dim1=0, dim2=1).movedim(-1, 0)   # [K-1, *rest]
    return E2[1:, 0] + E2[0, 1:] + diag
