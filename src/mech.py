"""src/mech.py — Reusable measurements for the modular-addition circuit."""
from __future__ import annotations

import math

import torch

from src.fourier import frequency_energy
from src.fourier2d import energy_classes, fourier2d_energy, per_frequency_score

R = "blocks.0."


def major_frequencies(model: torch.nn.Module, p: int, min_share: float) -> tuple[list[int], list[float]]:
    """Non-constant frequencies carrying >= min_share of W_E[0:p] energy."""
    E = frequency_energy(model.W_E[:p])
    frac = E / E.sum()
    ks = [k for k in range(1, len(E)) if frac[k].item() >= min_share]
    return ks, [frac[k].item() for k in ks]


def attention_from_eq(cache: dict) -> list[dict]:
    pat = cache[R + "attn.hook_pattern"][:, :, 2, :]           # [N, head, k_pos]
    return [{"head": h, "to_a": pat[:, h, 0].mean().item(), "to_b": pat[:, h, 1].mean().item(),
             "to_eq": pat[:, h, 2].mean().item()} for h in range(pat.shape[1])]


def neuron_structure(cache: dict, p: int, major: list[int]) -> dict:
    acts = cache[R + "mlp.hook_post"][:, 2, :].reshape(p, p, -1)  # [a, b, neuron]
    alive = acts.amax(dim=(0, 1)) >= 1e-8
    E2 = fourier2d_energy(acts)
    cls = energy_classes(E2)
    var = cls["total"] - cls["const"]
    alive = alive & (var > 0)
    tot = var[alive].sum()
    shares = {c: (cls[c][alive].sum() / tot).item() for c in ("a_only", "b_only", "same_freq", "cross")}
    dom_k = per_frequency_score(E2).argmax(dim=0) + 1
    n_alive = int(alive.sum())
    in_major = sum(int(k) in set(major) for k in dom_k[alive].tolist())
    return {"n_alive": n_alive, "frac_alive_dominant_in_major": in_major / max(n_alive, 1),
            "class_shares": shares}


def logit_fit(logits: torch.Tensor, p: int, freqs: list[int]) -> dict:
    if not freqs:
        return {"r2": float("nan"), "coef": {}}
    L = logits[:, -1, :p].to(torch.float64).reshape(p, p, p)
    L = L - L.mean(dim=2, keepdim=True)
    ar = torch.arange(p, dtype=torch.float64)
    s = ar[:, None, None] + ar[None, :, None] - ar[None, None, :]
    X = torch.stack([torch.cos(2 * math.pi * k * s / p).reshape(-1) for k in freqs], dim=1)
    y = L.reshape(-1, 1)
    coef = torch.linalg.lstsq(X, y).solution
    r2 = 1 - ((y - X @ coef) ** 2).sum() / (y ** 2).sum()
    return {"r2": r2.item(), "coef": {int(k): c.item() for k, c in zip(freqs, coef[:, 0])}}


def logit_components(model: torch.nn.Module, tokens: torch.Tensor, p: int) -> tuple[dict, torch.Tensor]:
    """Direct logit attribution at the '=' position, over the full (a, b) grid.
    Returns ({component: [p, p, p] float64 (a, b, c)}, total logits [p, p, p]).
    Components: constant parts (embed+pos of '=', MLP output bias) and the 2D Fourier
    classes of attn_out and of MLP neuron activations, each read out through W_U."""
    from src.fourier2d import CLASS_NAMES, project_out_classes

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    WU = model.W_U[:, :p].detach().to(torch.float64)                    # [d_model, p]
    W_out = model.blocks[0].mlp.W_out.detach().to(torch.float64)        # [d_mlp, d_model]
    b_out = model.blocks[0].mlp.b_out.detach().to(torch.float64)
    pos = 2
    grid = lambda x: x[:, pos].to(torch.float64).reshape(p, p, -1)     # [a, b, dim]

    comps = {}
    comps["const:embed+pos"] = (grid(cache["hook_embed"]) + grid(cache["hook_pos_embed"])) @ WU
    comps["const:mlp_bias"] = (b_out @ WU).expand(p, p, p).clone()
    attn, h = grid(cache[R + "hook_attn_out"]), grid(cache[R + "mlp.hook_post"])
    for cls in CLASS_NAMES:
        others = set(CLASS_NAMES) - {cls}
        comps[f"attn:{cls}"] = project_out_classes(attn, others) @ WU
        comps[f"mlp:{cls}"] = project_out_classes(h, others) @ W_out @ WU
    total = logits[:, -1, :p].to(torch.float64).reshape(p, p, p)
    return comps, total


def dla_metrics(L: torch.Tensor, p: int) -> dict:
    """L: [a, b, c]. Centered over c (softmax-invariant).
    norm2: squared Frobenius norm of the centered logits.
    margin: mean over (a, b) of L(a, b, c*) - mean_c L(a, b, c), c* = (a+b) mod p."""
    Lc = L - L.mean(dim=2, keepdim=True)
    ar = torch.arange(p)
    cstar = (ar[:, None] + ar[None, :]) % p
    margin = Lc.gather(2, cstar[:, :, None]).mean().item()
    return {"norm2": (Lc ** 2).sum().item(), "margin": margin}
