"""Calibration of the neuron-coherence measure. Run: uv run python -m tests.test_coherence"""
import torch

from src.data import modular_addition
from src.mech import coherence, dla_metrics, logit_components, mlp_class_coherence
from src.model import ModelConfig, Transformer

P = 113


def main() -> None:
    g = torch.Generator().manual_seed(0)
    h = torch.randn(1000, 1, generator=g, dtype=torch.float64)
    m = torch.randn(1, 7, generator=g, dtype=torch.float64)

    # 1. two identical neurons, opposite readout -> perfect cancellation
    k = coherence(torch.cat([h, h], 1), torch.cat([m, -m], 0))["kappa"]
    print(f"identical neurons, opposite readout: kappa {k:.2e}")
    assert abs(k) < 1e-12
    # 2. two identical neurons, identical readout -> kappa = 2
    k = coherence(torch.cat([h, h], 1), torch.cat([m, m], 0))["kappa"]
    print(f"identical neurons, identical readout: kappa {k:.6f}")
    assert abs(k - 2) < 1e-12
    # 3. many independent random neurons/readouts -> kappa ≈ 1
    k = coherence(torch.randn(5000, 200, generator=g, dtype=torch.float64),
                  torch.randn(200, 113, generator=g, dtype=torch.float64))["kappa"]
    print(f"independent random neurons: kappa {k:.3f} (expected ≈ 1)")
    assert 0.8 < k < 1.2

    # 4. consistency with DLA: 'joint' == norm² of the mlp:<class> logit component
    torch.manual_seed(0)
    model = Transformer(ModelConfig())
    tokens = modular_addition(P, 0.3, 598).tokens
    coh = mlp_class_coherence(model, tokens, P)
    comps, _ = logit_components(model, tokens, P)
    for cls in ("a_only", "b_only", "same_freq", "cross"):
        a, b = coh[cls]["joint"], dla_metrics(comps[f"mlp:{cls}"], P)["norm2"]
        print(f"{cls:9s} joint {a:.6e} vs DLA norm² {b:.6e}")
        assert abs(a - b) <= 1e-6 * max(abs(b), 1e-12)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
