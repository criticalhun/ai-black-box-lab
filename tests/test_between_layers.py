"""Calibration of the between-layer measurements. Run: uv run python -m tests.test_between_layers"""
import math

import torch

from src.mech import class_composition, harmonic_frequencies, univariate_spectrum

P = 113


def main() -> None:
    # 1. harmonics fold correctly: 2*60 = 120 = 7 (mod 113); 3*60 = 180 = 67 -> 113-67 = 46
    h = harmonic_frequencies([60], P)
    print(f"harmonics of 60: {h}")
    assert h == [7, 46]
    assert 5 not in harmonic_frequencies([5, 10], P)          # a major frequency is never a "harmonic"

    a = torch.arange(P, dtype=torch.float64)[:, None, None]
    b = torch.arange(P, dtype=torch.float64)[None, :, None]
    w = lambda k: 2 * math.pi * k / P
    zero = 0 * a * b

    # 2. class composition of known vectors (d = 2 dims)
    X = torch.cat([torch.cos(w(5) * a) + zero, torch.cos(w(5) * (a + b)) + zero], dim=2)
    c = class_composition(X, [5])
    print("composition:", {k: round(v, 4) for k, v in c.items()})
    assert abs(c["a_only_share"] - 0.5) < 1e-10 and abs(c["same_freq_share"] - 0.5) < 1e-10
    assert abs(c["major_share"] - 1.0) < 1e-10 and c["const_share"] < 1e-20

    # 3. ReLU of a pure cosine creates harmonics: energy also on 2k, not only on k
    X = torch.relu(torch.cos(w(5) * a)) + zero
    s = univariate_spectrum(X, [5], P)
    print("ReLU(cos 5a) spectrum:", {k: round(v, 4) for k, v in s.items()})
    assert s["major"] > 0.5 and s["harmonics"] > 0.05

    # 4. a pure cosine has no harmonics
    s = univariate_spectrum(torch.cos(w(5) * a) + zero, [5], P)
    assert abs(s["major"] - 1) < 1e-10 and s["harmonics"] < 1e-20

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
