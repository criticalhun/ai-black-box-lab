"""Calibration of 2D class projection + MLP activation hook.
Run: uv run python -m tests.test_class_ablation"""
import math

import torch

from src.data import modular_addition
from src.fourier2d import CLASS_NAMES, project_out_classes
from src.interventions import mlp_class_ablation_hook
from src.model import ModelConfig, Transformer

P = 113
HOOK = "blocks.0.mlp.hook_post"


def main() -> None:
    a = torch.arange(P, dtype=torch.float64)[:, None]
    b = torch.arange(P, dtype=torch.float64)[None, :]
    w = lambda k: 2 * math.pi * k / P
    same = torch.cos(w(5) * (a + b))
    aonly = torch.cos(w(3) * a) + 0 * b
    bonly = torch.sin(w(7) * b) + 0 * a
    cross = torch.cos(w(2) * a) * torch.cos(w(11) * b)
    const = 0.7 * torch.ones(P, P, dtype=torch.float64)
    f = same + aonly + bonly + cross + const

    # 1. removing nothing -> unchanged
    assert torch.allclose(project_out_classes(f, set()), f, atol=1e-12)
    # 2. each class removal removes exactly its part
    parts = {"same_freq": same, "a_only": aonly, "b_only": bonly, "cross": cross, "const": const}
    for cls, part in parts.items():
        err = (project_out_classes(f, {cls}) - (f - part)).abs().max().item()
        print(f"remove {cls:9s}: max error {err:.1e}")
        assert err < 1e-10
    # 3. complementary removals add up to the original (extra neuron dim)
    g = torch.Generator().manual_seed(0)
    M = torch.randn(P, P, 5, generator=g, dtype=torch.float64)
    S = {"a_only", "same_freq"}
    rest = set(CLASS_NAMES) - S
    assert torch.allclose(project_out_classes(M, S) + project_out_classes(M, rest), M, atol=1e-10)
    # 4. unknown class name fails loudly
    try:
        project_out_classes(M, {"typo"})
        raise AssertionError("unknown class should fail")
    except ValueError:
        pass

    # 5. hook on a model: no-op hook == original, only position 2 is touched
    torch.manual_seed(0)
    model = Transformer(ModelConfig())
    tokens = modular_addition(P, 0.3, 598).tokens
    with torch.no_grad():
        base, cache = model.run_with_cache(tokens)
        noop = model.run_with_hooks(tokens, [(HOOK, mlp_class_ablation_hook(P, set()))])
        diff = (noop - base).abs().max().item()
        print(f"no-op hook max |Δlogit|: {diff:.1e}")
        assert diff < 1e-4
        seen = {}

        def capture(act, name):
            seen["post"] = act.clone()

        model.run_with_hooks(tokens, [(HOOK, mlp_class_ablation_hook(P, {"same_freq"})), (HOOK, capture)])
        orig = cache[HOOK]
        assert torch.equal(seen["post"][:, :2, :], orig[:, :2, :]), "positions 0,1 changed"
        assert not torch.allclose(seen["post"][:, 2, :], orig[:, 2, :])
    # 6. wrong batch size fails loudly
    try:
        with torch.no_grad():
            model.run_with_hooks(tokens[:10], [(HOOK, mlp_class_ablation_hook(P, set()))])
        raise AssertionError("partial batch should fail")
    except ValueError:
        pass

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
