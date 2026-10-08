"""
EXP004 (EXPLORATORY): what do attention, MLP neurons and logits do?
No hypothesis, no verdict — output is for generating hypotheses (H004+),
which will be tested on FRESH seeds.

    uv run python experiments/EXP004/explore.py --config experiments/EXP004/config.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data import modular_addition  # noqa: E402
from src.fourier2d import energy_classes, fourier2d_energy, per_frequency_score  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402

R = "blocks.0."
CLASSES = ["a_only", "b_only", "same_freq", "cross"]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def attention_summary(cache: dict) -> list[dict]:
    pat = cache[R + "attn.hook_pattern"][:, :, 2, :]           # [N, head, k_pos] at "="
    return [{"head": h,
             "to_a_mean": pat[:, h, 0].mean().item(), "to_a_std": pat[:, h, 0].std().item(),
             "to_b_mean": pat[:, h, 1].mean().item(), "to_b_std": pat[:, h, 1].std().item(),
             "to_eq_mean": pat[:, h, 2].mean().item()} for h in range(pat.shape[1])]


def neuron_summary(cache: dict, p: int, key: set[int]) -> dict:
    acts = cache[R + "mlp.hook_post"][:, 2, :].reshape(p, p, -1)  # [a, b, neuron]
    dead = (acts.amax(dim=(0, 1)) < 1e-8)
    E2 = fourier2d_energy(acts)
    cls = energy_classes(E2)
    var = cls["total"] - cls["const"]                            # per-neuron non-constant energy
    alive = (~dead) & (var > 0)
    tot_var = var[alive].sum()
    class_share = {c: (cls[c][alive].sum() / tot_var).item() for c in CLASSES}
    score = per_frequency_score(E2)                              # [K-1, neuron]
    dom_k = score.argmax(dim=0) + 1
    dom_frac = score.max(dim=0).values / var.clamp_min(1e-30)
    key_idx = torch.tensor(sorted(key)) - 1
    key_share = (score[key_idx][:, alive].sum() / tot_var).item()
    by_freq = {}
    for k in sorted(set(dom_k[alive].tolist())):
        sel = alive & (dom_k == k)
        by_freq[int(k)] = {"n": int(sel.sum()), "mean_frac": dom_frac[sel].mean().item()}
    return {"n_neurons": int(acts.shape[-1]), "n_dead": int(dead.sum()),
            "class_share_of_variance": class_share, "key_freq_share_of_variance": key_share,
            "dominant_freq": by_freq}


def logit_fit(logits: torch.Tensor, p: int, key: list[int]) -> dict:
    L = logits[:, -1, :p].to(torch.float64).reshape(p, p, p)   # [a, b, c]
    L = L - L.mean(dim=2, keepdim=True)                          # softmax-invariant offset removed
    ar = torch.arange(p, dtype=torch.float64)
    s = ar[:, None, None] + ar[None, :, None] - ar[None, None, :]  # a + b - c
    X = torch.stack([torch.cos(2 * math.pi * k * s / p).reshape(-1) for k in key], dim=1)
    y = L.reshape(-1, 1)
    coef = torch.linalg.lstsq(X, y).solution
    r2 = 1 - ((y - X @ coef) ** 2).sum() / (y ** 2).sum()
    return {"r2": r2.item(), "coef": {int(k): c.item() for k, c in zip(key, coef[:, 0])}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    src = cfg["source"]
    assert cfg["evaluation"] == {"inputs": "all_pairs", "device": "cpu"}

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    exp1 = ROOT / "experiments" / src["experiment"]
    exp1_cfg = yaml.safe_load((exp1 / "config.yaml").read_text())
    model_cfg = ModelConfig.from_dict(exp1_cfg["model"])
    p = exp1_cfg["task"]["p"]
    tokens = modular_addition(p, exp1_cfg["task"]["train_fraction"], exp1_cfg["task"]["data_seed"]).tokens
    exp2 = json.loads((ROOT / "experiments" / cfg["key_frequencies_from"]
                       / "results" / "summary.json").read_text())

    def load(rdir: Path, name: str, expected_hash: str) -> Transformer:
        m = Transformer(model_cfg)
        m.load_state_dict(torch.load(rdir / "checkpoints" / name, map_location="cpu", weights_only=True))
        assert state_sha256(m) == expected_hash, f"{rdir.name}/{name}: checkpoint hash mismatch"
        return m

    def analyze(model: Transformer, key: list[int]) -> dict:
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        return {"attention": attention_summary(cache), "neurons": neuron_summary(cache, p, set(key)),
                "logit_fit": logit_fit(logits, p, key)}

    results = {}
    for run in src["runs"]:
        rdir = exp1 / "results" / run
        s1 = json.loads((rdir / "summary.json").read_text())
        key = sorted(exp2["runs"][run]["final"]["top_k_freqs"])
        results[run] = {
            "key_freqs": key,
            "final": analyze(load(rdir, src["checkpoint"], s1["final_state_sha256"]), key),
            "control": analyze(load(rdir, src["control_checkpoint"], s1["init_state_sha256"]), key),
        }

    print("=== EXP004 — EXPLORATORY (no hypothesis, no verdict) ===")
    print("Every number is shown for the FINAL model and for its own INIT (control).")
    for run, r in results.items():
        f, c = r["final"], r["control"]
        print(f"\n##### {run}  (EXP002 top-8: {r['key_freqs']})")
        print("attention from '=' (mean ± std over all inputs)        | init")
        for hf, hc in zip(f["attention"], c["attention"]):
            print(f"  head {hf['head']}: a {hf['to_a_mean']:.3f}±{hf['to_a_std']:.3f}  "
                  f"b {hf['to_b_mean']:.3f}±{hf['to_b_std']:.3f}  = {hf['to_eq_mean']:.3f}"
                  f"   | a {hc['to_a_mean']:.3f}  b {hc['to_b_mean']:.3f}  = {hc['to_eq_mean']:.3f}")
        nf, nc = f["neurons"], c["neurons"]
        print(f"MLP neurons at '=': dead {nf['n_dead']}/{nf['n_neurons']}   | init dead {nc['n_dead']}")
        print("  share of neuron variance:   final   init")
        for cl in CLASSES:
            print(f"    {cl:10s}                {nf['class_share_of_variance'][cl]:.3f}   "
                  f"{nc['class_share_of_variance'][cl]:.3f}")
        print(f"    top-8 freqs (k,0)+(0,k)+(k,k) {nf['key_freq_share_of_variance']:.3f}   "
              f"{nc['key_freq_share_of_variance']:.3f}")
        top = sorted(nf["dominant_freq"].items(), key=lambda kv: -kv[1]["n"])[:10]
        print("  final: neurons by dominant frequency (top 10): " + ", ".join(
            f"k={k}: {d['n']} (frac {d['mean_frac']:.2f})" for k, d in top))
        print(f"logits ~ Σ α_k cos(w_k (a+b−c)) over top-8:  R² final {f['logit_fit']['r2']:.4f}"
              f"   | init {c['logit_fit']['r2']:.4f}")
        print("  final α_k: " + ", ".join(f"{k}: {v:+.2f}" for k, v in f["logit_fit"]["coef"].items()))

    summary = {"experiment": cfg["experiment"], "hypothesis": None, "exploratory": True,
               "git_commit": git_commit(), "runs": results, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
