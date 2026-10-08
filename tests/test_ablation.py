"""Calibration of Fourier projection + embedding intervention.
Run: uv run python -m tests.test_ablation"""
import math

import torch

from src.fourier import basis_frequencies, frequency_energy, project_frequencies
from src.interventions import embed_keep_frequencies
from src.model import ModelConfig, Transformer
from src.repro import state_sha256

P = 113
ALL = set(range(57))


def main() -> None:
    g = torch.Generator().manual_seed(0)
    n = torch.arange(P, dtype=torch.float64)
    W = 0.05 * torch.randn(P, 128, generator=g, dtype=torch.float64)
    for k in (5, 17):
        ang = 2 * math.pi * k * n / P
        W += torch.outer(torch.cos(ang), torch.randn(128, generator=g, dtype=torch.float64))
        W += torch.outer(torch.sin(ang), torch.randn(128, generator=g, dtype=torch.float64))
    E = frequency_energy(W)

    bf = basis_frequencies(P).tolist()
    assert len(bf) == P and bf[:5] == [0, 1, 1, 2, 2] and bf[-1] == 56

    # 1. keep everything -> unchanged
    assert torch.allclose(project_frequencies(W, ALL), W, atol=1e-12)
    # 2. keep nothing -> zero
    assert project_frequencies(W, set()).abs().max().item() < 1e-12
    # 3. remove {5}: its energy vanishes, every other frequency untouched
    W5 = project_frequencies(W, ALL - {5})
    E5 = frequency_energy(W5)
    others = [k for k in range(57) if k != 5]
    print(f"energy at k=5: before {E[5].item():.3e}, after {E5[5].item():.3e}")
    assert E5[5].item() < 1e-20
    assert torch.allclose(E5[others], E[others], rtol=1e-10, atol=1e-20)
    # 4. idempotent
    assert torch.allclose(project_frequencies(W5, ALL - {5}), W5, atol=1e-12)
    # 5. complementary parts add up to the original
    S = {0, 5, 9, 30}
    assert torch.allclose(project_frequencies(W, S) + project_frequencies(W, ALL - S), W, atol=1e-12)

    # 6. model level: only W_E[0:113] changes; the original model is untouched
    torch.manual_seed(0)
    model = Transformer(ModelConfig())
    h0 = state_sha256(model)
    m2 = embed_keep_frequencies(model, ALL - {5})
    assert state_sha256(model) == h0, "original model was modified"
    sd1, sd2 = model.state_dict(), m2.state_dict()
    for name in sd1:
        if name == "W_E":
            assert torch.equal(sd1[name][113], sd2[name][113]), "'=' row changed"
            assert not torch.equal(sd1[name][:113], sd2[name][:113])
        else:
            assert torch.equal(sd1[name], sd2[name]), f"{name} changed"
    # 7. keep-all on a model = no-op up to float32 rounding
    assert torch.allclose(embed_keep_frequencies(model, ALL).W_E, model.W_E, atol=1e-6)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
