"""src/dynamics.py — Measurements across runs: frequency sets, alias margin, error offsets,
training instability, crossed-design variance decomposition. (EXP021–EXP024)"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from src.fourier import frequency_energy


# ---------- frequency sets ----------

def freq_set(W_E: torch.Tensor, p: int, min_share: float) -> tuple[list[int], list[float]]:
    """Non-constant frequencies carrying >= min_share of W_E[0:p] energy (same rule as mech.major_frequencies)."""
    E = frequency_energy(W_E[:p])
    frac = E / E.sum()
    ks = [k for k in range(1, len(E)) if frac[k].item() >= min_share]
    return ks, [frac[k].item() for k in ks]


def load_W_E(ckpt: Path) -> torch.Tensor:
    return torch.load(ckpt, map_location="cpu", weights_only=True)["W_E"]


def freq_trajectory(ckpt_dir: Path, p: int, min_share: float) -> list[dict]:
    """Frequency set at every saved checkpoint (W_E only)."""
    out = []
    for f in sorted(ckpt_dir.glob("step*.pt")):
        ks, sh = freq_set(load_W_E(f), p, min_share)
        out.append({"step": int(f.stem[4:]), "K": len(ks), "freqs": ks, "shares": sh})
    return out


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


# ---------- alias margin ----------

def alias_margin(freqs: list[int], p: int) -> dict:
    """Ideal-clock score of a wrong answer at offset d: S(d) = sum_k cos(2*pi*k*d/p); S(0) = K.
    margin = K - max_{d != 0} S(d); rel_margin = margin / K. Small rel_margin -> some wrong answer
    c = a+b-d is almost as good as the correct one."""
    K = len(freqs)
    if K == 0:
        return {"K": 0, "margin": float("nan"), "rel_margin": float("nan"), "top_offsets": []}
    d = torch.arange(1, p, dtype=torch.float64)
    S = sum(torch.cos(2 * math.pi * k * d / p) for k in freqs)
    order = torch.argsort(S, descending=True)
    top = [(int(d[i]), S[i].item()) for i in order[:4]]
    margin = K - S.max().item()
    return {"K": K, "margin": margin, "rel_margin": margin / K, "top_offsets": top}


# ---------- error offsets ----------

def error_offsets(pred: torch.Tensor, tokens: torch.Tensor, labels: torch.Tensor, p: int) -> dict:
    """For wrong predictions: delta = (pred - (a+b)) mod p. Returns counts per delta."""
    wrong = pred != labels
    delta = (pred[wrong] - (tokens[wrong, 0] + tokens[wrong, 1])) % p
    counts = torch.bincount(delta, minlength=p)
    return {"n": int(labels.numel()), "n_wrong": int(wrong.sum()),
            "counts": {int(k): int(c) for k, c in enumerate(counts.tolist()) if c > 0}}


def share_in(counts: dict, offsets: set[int]) -> float:
    n = sum(counts.values())
    return sum(c for k, c in counts.items() if k in offsets) / n if n else float("nan")


# ---------- training instability ----------

def read_metrics(path: Path) -> list[dict]:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def instability(rows: list[dict], loss_threshold: float, buffer_steps: int) -> dict:
    """Events = maximal runs of consecutive logged rows with train_loss >= threshold (NaN/inf count),
    counted only after (first step with train_acc >= 0.99) + buffer_steps."""
    first = next((r["step"] for r in rows if r["train_acc"] >= 0.99), None)
    if first is None:
        return {"memorized_at": None, "events": None, "max_train_loss": None, "frac_rows_above": None}
    start = first + buffer_steps
    tail = [r for r in rows if r["step"] >= start]
    bad = [not (r["train_loss"] < loss_threshold) for r in tail]       # NaN -> bad
    events = sum(1 for i, b in enumerate(bad) if b and (i == 0 or not bad[i - 1]))
    losses = [r["train_loss"] for r in tail]
    mx = max(losses, key=lambda x: math.inf if not math.isfinite(x) else x) if losses else None
    return {"memorized_at": first, "events": events, "max_train_loss": mx,
            "frac_rows_above": sum(bad) / len(bad) if bad else None}


# ---------- crossed design ----------

def two_way_decomposition(Y: list[list[float]]) -> dict:
    """Additive two-way decomposition (rows = data_seed, cols = model seed), no replication.
    Returns sums of squares and their shares of the total. Descriptive only (tiny df)."""
    t = torch.tensor(Y, dtype=torch.float64)
    g = t.mean()
    r = t.mean(dim=1, keepdim=True) - g
    c = t.mean(dim=0, keepdim=True) - g
    res = t - g - r - c
    ss = {"rows": (r ** 2).sum().item() * t.shape[1], "cols": (c ** 2).sum().item() * t.shape[0],
          "residual": (res ** 2).sum().item()}
    tot = ((t - g) ** 2).sum().item()
    return {"ss": ss, "ss_total": tot,
            "share": {k: (v / tot if tot > 0 else float("nan")) for k, v in ss.items()}}


def early_topk_jaccard(traj: list[dict], ckpt_dir: Path, p: int, steps: list[int]) -> dict:
    """Jaccard between the FINAL frequency set (size K) and the top-K frequencies by W_E energy
    at earlier checkpoints — how early is the final set already visible?"""
    final = traj[-1]["freqs"]
    K = len(final)
    out = {}
    for st in steps:
        f = ckpt_dir / f"step{st:06d}.pt"
        if K == 0 or not f.exists():
            out[st] = None
            continue
        E = frequency_energy(load_W_E(f)[:p])[1:]
        top = (torch.argsort(E, descending=True)[:K] + 1).tolist()
        out[st] = jaccard(top, final)
    return out


def k_drops(traj: list[dict]) -> list[dict]:
    """Checkpoints where the number of major frequencies decreased."""
    return [{"step": b["step"], "K_from": a["K"], "K_to": b["K"], "lost": sorted(set(a["freqs"]) - set(b["freqs"]))}
            for a, b in zip(traj, traj[1:]) if b["K"] < a["K"]]
