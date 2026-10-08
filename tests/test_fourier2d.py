"""Calibration of src/fourier2d.py on functions with KNOWN structure.
Run: uv run python -m tests.test_fourier2d"""
import math

import torch

from src.fourier2d import energy_classes, fourier2d_energy, per_frequency_score

P = 113


def main() -> None:
    a = torch.arange(P, dtype=torch.float64)[:, None]
    b = torch.arange(P, dtype=torch.float64)[None, :]
    w = lambda k: 2 * math.pi * k / P

    def share(f, cls):
        c = energy_classes(fourier2d_energy(f))
        return (c[cls] / c["total"]).item()

    # 1. Parseval on a random function with an extra (neuron) dimension
    g = torch.Generator().manual_seed(0)
    M = torch.randn(P, P, 7, generator=g, dtype=torch.float64)
    E2 = fourier2d_energy(M)
    assert E2.shape == (57, 57, 7)
    assert torch.allclose(E2.sum((0, 1)), (M ** 2).sum((0, 1)), rtol=1e-10)

    # 2. known cases: each must land 100% in its own class
    cases = {
        "cos(w5 (a+b)) -> same_freq": (torch.cos(w(5) * (a + b)), "same_freq"),
        "cos(w5 a)     -> a_only":    (torch.cos(w(5) * a) + 0 * b, "a_only"),
        "sin(w9 b)     -> b_only":    (torch.sin(w(9) * b) + 0 * a, "b_only"),
        "cos(w5 a)cos(w17 b) -> cross": (torch.cos(w(5) * a) * torch.cos(w(17) * b), "cross"),
        "constant      -> const":     (torch.ones(P, P, dtype=torch.float64), "const"),
    }
    for name, (f, cls) in cases.items():
        s = share(f, cls)
        print(f"{name:32s} share {s:.6f}")
        assert s > 1 - 1e-10, (name, s)

    # 3. per-frequency score finds the right frequency
    f = torch.cos(w(5) * (a + b)) + 0.5 * torch.cos(w(17) * a)
    score = per_frequency_score(fourier2d_energy(f))
    assert sorted((torch.topk(score, 2).indices + 1).tolist()) == [5, 17]
    assert score.argmax().item() + 1 == 5

    # 4. random noise: same-frequency share is small (~ 56*4/113^2)
    s = share(torch.randn(P, P, generator=g, dtype=torch.float64), "same_freq")
    print(f"random noise same_freq share {s:.3f} (expected ~ {56 * 4 / P**2:.3f})")
    assert s < 0.05

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
