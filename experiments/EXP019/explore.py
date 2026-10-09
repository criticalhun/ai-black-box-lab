"""
EXP019 (EXPLORATORY): what does layer 0 of the 2-layer model prepare for layer 1?
Attention patterns per layer, the 2D Fourier make-up of what each component writes,
and where the single-number (univariate) energy of MLP0's output sits (major freqs vs harmonics).
No hypothesis, no verdict.

    uv run python experiments/EXP019/explore.py --config experiments/EXP019/config.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data import modular_addition  # noqa: E402
from src.mech import attention_at, class_composition, major_frequencies, univariate_spectrum  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402

# (label, hook name, position)
SITES = [
    ("attn0 out @'='", "blocks.0.hook_attn_out", 2),
    ("mlp0 out @'='", "blocks.0.hook_mlp_out", 2),
    ("mlp0 out @b", "blocks.0.hook_mlp_out", 1),
    ("resid after L0 @'='", "blocks.0.hook_resid_post", 2),
    ("attn1 out @'='", "blocks.1.hook_attn_out", 2),
    ("mlp1 out @'='", "blocks.1.hook_mlp_out", 2),
]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


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

    exp = ROOT / "experiments" / src["experiment"]
    exp_cfg = yaml.safe_load((exp / "config.yaml").read_text())
    model_cfg = ModelConfig.from_dict(exp_cfg["model"])
    assert model_cfg.n_layers == 2
    task = exp_cfg["task"]
    p = task["p"]
    tokens = modular_addition(p, task["train_fraction"], task["data_seed"]).tokens

    def load(rdir: Path, name: str, expected: str) -> Transformer:
        m = Transformer(model_cfg)
        m.load_state_dict(torch.load(rdir / "checkpoints" / name, map_location="cpu", weights_only=True))
        assert state_sha256(m) == expected, f"{rdir.name}/{name}: checkpoint hash mismatch"
        return m

    grid = lambda cache, hook, pos: cache[hook][:, pos].to(torch.float64).reshape(p, p, -1)

    results = {}
    for run in src["runs"]:
        rdir = exp / "results" / run
        s = json.loads((rdir / "summary.json").read_text())
        final = load(rdir, src["checkpoint"], s["final_state_sha256"])
        init = load(rdir, src["control_checkpoint"], s["init_state_sha256"])
        major, major_share = major_frequencies(final, p, cfg["major_frequency_min_share"])
        with torch.no_grad():
            _, cf = final.run_with_cache(tokens)
            _, ci = init.run_with_cache(tokens)
        results[run] = {
            "major_freqs": major, "major_share": major_share,
            "attention": {"L0 from '='": attention_at(cf, 0, 2), "L0 from b": attention_at(cf, 0, 1),
                          "L1 from '='": attention_at(cf, 1, 2)},
            "composition": {lab: {"final": class_composition(grid(cf, hk, q), major),
                                  "init": class_composition(grid(ci, hk, q), major)} for lab, hk, q in SITES},
            "mlp0_univariate_spectrum": {
                "input (resid_mid0) @'='": univariate_spectrum(grid(cf, "blocks.0.hook_resid_mid", 2), major, p),
                "output (mlp0 out) @'='": univariate_spectrum(grid(cf, "blocks.0.hook_mlp_out", 2), major, p),
                "init output (mlp0 out) @'='": univariate_spectrum(grid(ci, "blocks.0.hook_mlp_out", 2), major, p),
            },
        }

    print("=== EXP019 — EXPLORATORY: what does layer 0 prepare? (no verdict) ===")
    pos_name = ["a", "b", "="]
    for run, r in results.items():
        print(f"\n##### {run}   major freqs: " + ", ".join(
            f"{k} ({v:.2f})" for k, v in zip(r["major_freqs"], r["major_share"])))
        print("  attention (mean over all inputs):")
        for lab, heads in r["attention"].items():
            print(f"    {lab:12s} " + " | ".join(
                f"h{h['head']} " + " ".join(f"{pos_name[k]} {v:.2f}" for k, v in enumerate(h["to"]))
                for h in heads))
        print("  2D Fourier make-up (shares of non-constant energy; final  [init]):")
        print(f"    {'site':22s} {'a_only':>13s} {'b_only':>13s} {'same_freq':>13s} {'cross':>13s} {'on major':>13s}")
        for lab, c in r["composition"].items():
            f_, i_ = c["final"], c["init"]
            cell = lambda k: f"{f_[k]:.3f} [{i_[k]:.3f}]"
            print(f"    {lab:22s} {cell('a_only_share'):>13s} {cell('b_only_share'):>13s} "
                  f"{cell('same_freq_share'):>13s} {cell('cross_share'):>13s} {cell('major_share'):>13s}")
        print("  single-number energy of MLP0 @'=': share at major freqs / their 2nd-3rd harmonics / other")
        for lab, sp in r["mlp0_univariate_spectrum"].items():
            print(f"    {lab:28s} major {sp['major']:.3f} | harmonics {sp['harmonics']:.3f} "
                  f"({sp['n_harmonics']} freqs) | other {sp['other']:.3f}")

    summary = {"experiment": cfg["experiment"], "hypothesis": None, "exploratory": True,
               "git_commit": git_commit(), "runs": results, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
