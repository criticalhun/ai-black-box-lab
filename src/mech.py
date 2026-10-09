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


def coherence(H: torch.Tensor, M: torch.Tensor) -> dict:
    """H: [N, n] activations of n neurons over N inputs; M: [n, p] readout of each neuron
    to the p logits. Contribution of neuron j: L_j(x, c) = H[x, j] * Mc[j, c], Mc = M centered over c.
    joint = ||Σ_j L_j||²,  individual = Σ_j ||L_j||²,  kappa = joint / individual.
    kappa ≈ 1: incoherent sum; << 1: neurons cancel each other; > 1: they reinforce."""
    H = H.to(torch.float64)
    Mc = M.to(torch.float64) - M.to(torch.float64).mean(dim=1, keepdim=True)
    Gh, Gm = H.T @ H, Mc @ Mc.T
    joint = (Gh * Gm).sum().item()
    indiv = (torch.diagonal(Gh) * torch.diagonal(Gm)).sum().item()
    return {"joint": joint, "individual": indiv, "kappa": joint / indiv if indiv > 0 else float("nan")}


COHERENCE_GROUPS = {"a_only": {"a_only"}, "b_only": {"b_only"}, "univariate": {"a_only", "b_only"},
                    "same_freq": {"same_freq"}, "cross": {"cross"}}


def mlp_class_coherence(model: torch.nn.Module, tokens: torch.Tensor, p: int) -> dict:
    """Coherence of the MLP neurons' direct logit contributions, per 2D Fourier class."""
    from src.fourier2d import CLASS_NAMES, project_out_classes

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    h = cache[R + "mlp.hook_post"][:, 2].to(torch.float64).reshape(p, p, -1)
    M = (model.blocks[0].mlp.W_out.detach().to(torch.float64)
         @ model.W_U[:, :p].detach().to(torch.float64))                     # [d_mlp, p]
    out = {}
    for g, keep in COHERENCE_GROUPS.items():
        hX = project_out_classes(h, set(CLASS_NAMES) - keep).reshape(p * p, -1)
        out[g] = coherence(hX, M)
    return out


# ---------- layer-aware versions (n_layers >= 1) ----------

def neuron_structure_at(cache: dict, p: int, major: list[int], layer: int, pos: int) -> dict:
    """Like neuron_structure, for any layer and token position."""
    acts = cache[f"blocks.{layer}.mlp.hook_post"][:, pos, :].reshape(p, p, -1)
    alive = acts.amax(dim=(0, 1)) >= 1e-8
    E2 = fourier2d_energy(acts)
    cls = energy_classes(E2)
    var = cls["total"] - cls["const"]
    alive = alive & (var > 0)
    tot = var[alive].sum()
    if not bool(alive.any()):
        return {"n_alive": 0, "frac_alive_dominant_in_major": float("nan"),
                "class_shares": {c: float("nan") for c in ("a_only", "b_only", "same_freq", "cross")}}
    shares = {c: (cls[c][alive].sum() / tot).item() for c in ("a_only", "b_only", "same_freq", "cross")}
    dom_k = per_frequency_score(E2).argmax(dim=0) + 1
    n_alive = int(alive.sum())
    in_major = sum(int(k) in set(major) for k in dom_k[alive].tolist())
    return {"n_alive": n_alive, "frac_alive_dominant_in_major": in_major / n_alive,
            "class_shares": shares}


def residual_dla(model: torch.nn.Module, tokens: torch.Tensor, p: int) -> dict:
    """Direct logit attribution at '=' for a model of any depth: embed+pos, and each
    layer's attn_out / mlp_out. Returns {component: dla_metrics} + 'total'.
    Note: DIRECT path only — what a layer contributes via later layers is not counted."""
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    WU = model.W_U[:, :p].detach().to(torch.float64)
    grid = lambda x: x[:, 2].to(torch.float64).reshape(p, p, -1)
    comps = {"embed+pos": (grid(cache["hook_embed"]) + grid(cache["hook_pos_embed"])) @ WU}
    for L in range(len(model.blocks)):
        comps[f"attn{L}"] = grid(cache[f"blocks.{L}.hook_attn_out"]) @ WU
        comps[f"mlp{L}"] = grid(cache[f"blocks.{L}.hook_mlp_out"]) @ WU
    out = {k: dla_metrics(v, p) for k, v in comps.items()}
    out["total"] = dla_metrics(logits[:, -1, :p].to(torch.float64).reshape(p, p, p), p)
    return out


# ---------- what flows between layers ----------

def attention_at(cache: dict, layer: int, pos: int = 2) -> list[dict]:
    """Mean attention from query position `pos` to every key position, per head, for a layer."""
    pat = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, pos, :pos + 1]   # [N, head, k]
    return [{"head": h, "to": [pat[:, h, k].mean().item() for k in range(pat.shape[2])],
             "std": [pat[:, h, k].std().item() for k in range(pat.shape[2])]} for h in range(pat.shape[1])]


def harmonic_frequencies(major: list[int], p: int, orders: tuple[int, ...] = (2, 3)) -> list[int]:
    """Frequencies m*k (folded into 1..(p-1)/2) for k in `major`, m in `orders`, excluding `major` itself."""
    out = set()
    for k in major:
        for m in orders:
            r = (m * k) % p
            f = min(r, p - r)
            if f != 0 and f not in major:
                out.add(f)
    return sorted(out)


def class_composition(X: torch.Tensor, major: list[int]) -> dict:
    """X: [p, p, d] (a vector per (a, b)). Shares of its NON-constant energy per 2D Fourier class,
    the constant share of the total, and the share of non-constant energy at major frequencies
    (E(k,0) + E(0,k) + E(k,k) summed over k in major)."""
    E2 = fourier2d_energy(X).sum(dim=tuple(range(2, X.dim())))          # [K, K]
    c = energy_classes(E2)
    var = (c["total"] - c["const"]).item()
    score = per_frequency_score(E2)                                      # [K-1]
    major_part = sum(score[k - 1].item() for k in major)
    return {"const_share": (c["const"] / c["total"]).item(),
            **{f"{k}_share": c[k].item() / var for k in ("a_only", "b_only", "same_freq", "cross")},
            "major_share": major_part / var}


def univariate_spectrum(X: torch.Tensor, major: list[int], p: int) -> dict:
    """Where the a-only and b-only energy of X [p, p, d] sits: at major frequencies,
    at their 2nd/3rd harmonics, or elsewhere (shares of the a_only + b_only energy)."""
    E2 = fourier2d_energy(X).sum(dim=tuple(range(2, X.dim())))
    uni = E2[1:, 0] + E2[0, 1:]                                          # per k = 1..K-1
    tot = uni.sum().item()
    harm = harmonic_frequencies(major, p)
    at = lambda ks: sum(uni[k - 1].item() for k in ks) / tot
    return {"major": at(major), "harmonics": at(harm), "other": 1 - at(major) - at(harm),
            "n_harmonics": len(harm)}
