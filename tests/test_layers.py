"""Calibration of the layer-aware tools on a 2-layer model.
Run: uv run python -m tests.test_layers"""
import torch

from src.data import modular_addition
from src.interventions import mean_ablation_hook
from src.mech import neuron_structure, neuron_structure_at, residual_dla
from src.model import ModelConfig, Transformer

P = 113


def main() -> None:
    tokens = modular_addition(P, 0.3, 598).tokens

    # 1. on a 1-layer model the layer-aware function equals the original one
    torch.manual_seed(0)
    m1 = Transformer(ModelConfig())
    with torch.no_grad():
        _, c1 = m1.run_with_cache(tokens)
    a, b = neuron_structure(c1, P, [9, 19]), neuron_structure_at(c1, P, [9, 19], layer=0, pos=2)
    assert a == b, (a, b)
    print("neuron_structure_at(layer 0, pos 2) == neuron_structure on 1-layer model")

    # 2. 2-layer model: DLA component margins add up exactly to the total margin
    torch.manual_seed(0)
    m2 = Transformer(ModelConfig(n_layers=2))
    d = residual_dla(m2, tokens, P)
    assert set(d) == {"embed+pos", "attn0", "mlp0", "attn1", "mlp1", "total"}
    s = sum(v["margin"] for k, v in d.items() if k != "total")
    print(f"2-layer: Σ component margins {s:+.6e} vs total {d['total']['margin']:+.6e}")
    assert abs(s - d["total"]["margin"]) < 1e-6

    # 3. mean ablation: output identical across the batch at every position, mean preserved
    seen = {}

    def capture(act, name):
        seen["x"] = act.clone()

    with torch.no_grad():
        _, cache = m2.run_with_cache(tokens)
        m2.run_with_hooks(tokens, [("blocks.0.hook_mlp_out", mean_ablation_hook()),
                                   ("blocks.0.hook_mlp_out", capture)])
    x, orig = seen["x"], cache["blocks.0.hook_mlp_out"]
    assert torch.allclose(x, x[:1].expand_as(x)), "not constant over the batch"
    assert torch.allclose(x[0], orig.mean(0), atol=1e-6), "mean not preserved"
    print("mean ablation: constant over inputs, equal to the mean activation")

    # 4. structure works at position 1 of layer 0 (b's position sees a and b)
    with torch.no_grad():
        _, c2 = m2.run_with_cache(tokens)
    r = neuron_structure_at(c2, P, [9], layer=0, pos=1)
    print(f"layer 0, pos 1: alive {r['n_alive']}, same_freq share {r['class_shares']['same_freq']:.4f}")
    assert r["n_alive"] > 0

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
