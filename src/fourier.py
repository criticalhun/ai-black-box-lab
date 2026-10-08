"""src/fourier.py — Real orthonormal Fourier basis on Z_p and frequency-energy analysis."""
from __future__ import annotations

import math

import torch


def fourier_basis(p: int) -> tuple[torch.Tensor, list[str]]:
    """Orthonormal real Fourier basis on Z_p (p odd), float64.
    Rows: const, cos1, sin1, cos2, sin2, ..., cos((p-1)/2), sin((p-1)/2)."""
    if p % 2 == 0:
        raise ValueError("p must be odd")
    n = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64) / math.sqrt(p)]
    labels = ["const"]
    for k in range(1, (p - 1) // 2 + 1):
        ang = 2 * math.pi * k * n / p
        rows += [torch.cos(ang) * math.sqrt(2 / p), torch.sin(ang) * math.sqrt(2 / p)]
        labels += [f"cos{k}", f"sin{k}"]
    return torch.stack(rows), labels


def frequency_energy(W: torch.Tensor) -> torch.Tensor:
    """W: [p, d], row n = vector of number n.
    Returns energy per frequency k = 0..(p-1)//2 (k=0 is the constant).
    Parseval: result.sum() == ||W||_F^2."""
    F, _ = fourier_basis(W.shape[0])
    C = F @ W.detach().cpu().to(torch.float64)
    e = (C ** 2).sum(dim=1)
    return torch.cat([e[:1], e[1::2] + e[2::2]])


def top_k_share(energy: torch.Tensor, k: int) -> tuple[float, list[int]]:
    """Share of TOTAL energy (incl. constant) carried by the k strongest
    non-constant frequencies. Returns (share, sorted frequency list)."""
    vals, idx = torch.topk(energy[1:], k)
    return (vals.sum() / energy.sum()).item(), sorted((idx + 1).tolist())
