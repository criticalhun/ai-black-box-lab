"""Calibration of src/dynamics.py on known answers. Run: uv run python -m tests.test_dynamics"""
import itertools
import math

import torch

from src.dynamics import (alias_margin, error_offsets, freq_set, instability, jaccard, share_in,
                          two_way_decomposition)
from src.fourier import fourier_basis

P = 113


def ideal_clock_preds(freqs, noise, seed=0):
    """Predictions of an ideal Fourier-clock model: logit(c) = sum_k cos(w_k (a+b-c)) + Gaussian noise."""
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    c = torch.arange(P)
    diff = (a + b)[:, None] - c[None, :]
    L = sum(torch.cos(2 * math.pi * k * diff / P) for k in freqs)
    L = L + noise * torch.randn(L.shape, generator=g, dtype=torch.float64)
    tokens = torch.stack([a, b, torch.full_like(a, P)], dim=1)
    return L.argmax(-1), tokens, (a + b) % P


def main() -> None:
    # 1. alias margin: the seed3 set {28, 43} aliases at d = 8 (28*8 = -2, 43*8 = 5 mod 113)
    m = alias_margin([28, 43], P)
    print("alias {28,43}:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    assert {m["top_offsets"][0][0], m["top_offsets"][1][0]} == {8, 105}
    assert m["rel_margin"] < 0.03

    # 2. theory used in H026/H027: EVERY 2-frequency set has rel_margin < 0.10 for p = 113
    worst = max(alias_margin([i, j], P)["rel_margin"] for i, j in itertools.combinations(range(1, 57), 2))
    print(f"max rel_margin over all 2-frequency sets: {worst:.4f}")
    assert worst < 0.10

    # 3. error offsets of an ideal noisy clock: errors concentrate on the alias offsets
    pred, tok, lab = ideal_clock_preds([28, 43], noise=0.03)
    e = error_offsets(pred, tok, lab, P)
    s8 = share_in(e["counts"], {8, 105})
    print(f"ideal clock {{28,43}} + noise: wrong {e['n_wrong']}, share at +-8 = {s8:.3f}")
    assert e["n_wrong"] > 50 and s8 > 0.9
    # control: a 3-frequency set with a large margin makes no errors at the same noise
    pred, tok, lab = ideal_clock_preds([2, 11, 19], noise=0.03)
    e = error_offsets(pred, tok, lab, P)
    print(f"ideal clock {{2,11,19}} + noise: wrong {e['n_wrong']}")
    assert e["n_wrong"] == 0
    # exact bookkeeping on a hand-made case
    tok = torch.tensor([[1, 2, P], [10, 20, P], [0, 0, P]])
    lab = torch.tensor([3, 30, 0])
    e = error_offsets(torch.tensor([3, 22, 105]), tok, lab, P)
    assert e == {"n": 3, "n_wrong": 2, "counts": {105: 2}}, e

    # 4. freq_set recovers planted frequencies
    F, _ = fourier_basis(P)
    W = 3 * F[2 * 7 - 1][:, None] * torch.ones(1, 4) + 2 * F[2 * 30][:, None] * torch.ones(1, 4)
    ks, sh = freq_set(W, P, 0.10)
    print("planted 7 (cos) and 30 (sin):", ks, [round(x, 3) for x in sh])
    assert ks == [7, 30] and abs(sh[0] - 9 / 13) < 1e-10

    # 5. jaccard
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5 and jaccard([1], [2]) == 0.0

    # 6. instability events (NaN counts as unstable; rows before memorization + buffer ignored)
    rows = [{"step": s, "train_acc": 1.0 if s >= 200 else 0.5, "train_loss": 1e-7} for s in range(0, 5000, 100)]
    for s, v in ((100, 5.0), (300, 1.0), (2000, 1e-2), (2100, 1e-1), (3000, float("nan")), (4000, 5e-4)):
        rows[s // 100]["train_loss"] = v
    r = instability(rows, loss_threshold=1e-3, buffer_steps=1000)
    print("instability:", r)
    assert r["memorized_at"] == 200 and r["events"] == 2   # {2000,2100} and {3000}; 300 is in buffer, 4000 below thr

    # 7. two-way decomposition
    d = two_way_decomposition([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert abs(d["share"]["rows"] - 1) < 1e-12 and d["share"]["cols"] < 1e-12
    d = two_way_decomposition([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert abs(d["share"]["cols"] - 1) < 1e-12

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
