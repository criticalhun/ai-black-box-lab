"""Calibration of direct logit attribution: components must add up EXACTLY to the logits.
Run: uv run python -m tests.test_dla"""
import torch

from src.data import modular_addition
from src.mech import dla_metrics, logit_components
from src.model import ModelConfig, Transformer

P = 113


def main() -> None:
    torch.manual_seed(0)
    model = Transformer(ModelConfig())
    tokens = modular_addition(P, 0.3, 598).tokens
    comps, total = logit_components(model, tokens, P)
    assert len(comps) == 12, list(comps)
    s = sum(comps.values())
    err = (s - total).abs().max().item()
    print(f"components: {len(comps)} | max |Σ components − logits|: {err:.1e}")
    assert err < 1e-4

    # margins are linear: they add up exactly as well
    m_sum = sum(dla_metrics(L, P)["margin"] for L in comps.values())
    m_tot = dla_metrics(total, P)["margin"]
    print(f"Σ margins {m_sum:+.6f} vs total margin {m_tot:+.6f}")
    assert abs(m_sum - m_tot) < 1e-6

    # a component constant in (a, b) carries no margin
    assert abs(dla_metrics(comps["const:mlp_bias"], P)["margin"]) < 1e-12
    # known case: logit exactly 1 at c* = (a+b) mod p, 0 elsewhere -> margin = 1 - 1/p
    ar = torch.arange(P)
    onehot = torch.zeros(P, P, P, dtype=torch.float64)
    onehot[ar[:, None], ar[None, :], (ar[:, None] + ar[None, :]) % P] = 1.0
    m = dla_metrics(onehot, P)["margin"]
    print(f"one-hot correct-answer margin {m:.6f} (expected {1 - 1 / P:.6f})")
    assert abs(m - (1 - 1 / P)) < 1e-12

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
