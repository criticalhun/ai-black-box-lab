"""Calibration of src/fourier.py on cases with KNOWN answers.
Run: uv run python -m tests.test_fourier"""
import math

import torch

from src.fourier import fourier_basis, frequency_energy, top_k_share

P = 113


def rand(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def main() -> None:
    F, labels = fourier_basis(P)
    assert F.shape == (P, P) and len(labels) == P
    err = (F @ F.T - torch.eye(P, dtype=torch.float64)).abs().max().item()
    print(f"orthonormality error: {err:.1e}")
    assert err < 1e-12

    g = torch.Generator().manual_seed(0)

    # 1. Parseval: energy is preserved
    W = rand(g, P, 128)
    E = frequency_energy(W)
    assert E.shape == (57,)
    assert math.isclose(E.sum().item(), (W ** 2).sum().item(), rel_tol=1e-10)

    # 2. "no structure" baseline: random Gaussian matrices
    shares = [top_k_share(frequency_energy(rand(g, P, 128)), 8)[0] for _ in range(50)]
    m = sum(shares) / len(shares)
    print(f"random W: top-8 share mean {m:.3f} (min {min(shares):.3f}, max {max(shares):.3f})")
    assert 0.12 < m < 0.22

    # 3. known structure: only frequencies 5 and 17 (+ small noise)
    n = torch.arange(P, dtype=torch.float64)
    W = 0.05 * rand(g, P, 128)
    for k in (5, 17):
        ang = 2 * math.pi * k * n / P
        W += torch.outer(torch.cos(ang), rand(g, 128)) + torch.outer(torch.sin(ang), rand(g, 128))
    share2, freqs2 = top_k_share(frequency_energy(W), 2)
    print(f"synthetic {{5,17}}: top-2 freqs {freqs2}, share {share2:.3f}")
    assert freqs2 == [5, 17] and share2 > 0.95

    # 4. a constant matrix has no non-constant frequency content
    E = frequency_energy(torch.ones(P, 128, dtype=torch.float64))
    assert (E[1:].sum() / E.sum()).item() < 1e-12

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
