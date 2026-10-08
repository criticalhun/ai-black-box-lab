"""
EXP009 (EXPLORATORY): direct logit attribution — which components of the residual
stream at '=' actually move the logits toward the correct answer?
No hypothesis, no verdict.

    uv run python experiments/EXP009/explore.py --config experiments/EXP009/config.yaml
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
from src.mech import dla_metrics, logit_components  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402

GROUPS = {
    "constant": ["const:embed+pos", "const:mlp_bias", "attn:const", "mlp:const"],
    "univariate": ["attn:a_only", "attn:b_only", "mlp:a_only", "mlp:b_only"],
    "same_freq": ["attn:same_freq", "mlp:same_freq"],
    "cross": ["attn:cross", "mlp:cross"],
}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def analyze(model: Transformer, tokens: torch.Tensor, p: int) -> dict:
    comps, total = logit_components(model, tokens, p)
    tot = dla_metrics(total, p)
    per = {k: dla_metrics(L, p) for k, L in comps.items()}
    groups = {}
    for g, keys in GROUPS.items():
        joint = dla_metrics(sum(comps[k] for k in keys), p)
        groups[g] = {"norm2": joint["norm2"], "margin": joint["margin"],
                     "sum_individual_norm2": sum(per[k]["norm2"] for k in keys)}
    return {"total": tot, "components": per, "groups": groups}


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
    task = exp_cfg["task"]
    p = task["p"]
    tokens = modular_addition(p, task["train_fraction"], task["data_seed"]).tokens

    def load(rdir: Path, name: str, expected: str) -> Transformer:
        m = Transformer(model_cfg)
        m.load_state_dict(torch.load(rdir / "checkpoints" / name, map_location="cpu", weights_only=True))
        assert state_sha256(m) == expected, f"{rdir.name}/{name}: checkpoint hash mismatch"
        return m

    results = {}
    for run in src["runs"]:
        rdir = exp / "results" / run
        s = json.loads((rdir / "summary.json").read_text())
        results[run] = {
            "final": analyze(load(rdir, src["checkpoint"], s["final_state_sha256"]), tokens, p),
            "control": analyze(load(rdir, src["control_checkpoint"], s["init_state_sha256"]), tokens, p),
        }

    print("=== EXP009 — EXPLORATORY: direct logit attribution at '=' (no verdict) ===")
    print("norm share = ||centered component||² / ||centered total logits||²  (shares need NOT sum to 1:")
    print("             components can cancel each other)")
    print("margin     = mean over (a,b) of [logit(correct c) − mean_c logit]; margins DO add up exactly")
    for run, r in results.items():
        f, c = r["final"], r["control"]
        T, Tm = f["total"]["norm2"], f["total"]["margin"]
        print(f"\n##### {run}   total margin: final {Tm:+.2f} | init {c['total']['margin']:+.4f}")
        print(f"  {'component':18s} {'norm share':>10s} {'margin':>9s} {'margin %':>9s}")
        for k, m in f["components"].items():
            print(f"  {k:18s} {m['norm2'] / T:10.4f} {m['margin']:+9.2f} {100 * m['margin'] / Tm:8.1f}%")
        print(f"  {'GROUP':18s} {'joint share':>10s} {'margin':>9s} {'margin %':>9s}  {'Σ indiv. share':>14s}")
        for g, m in f["groups"].items():
            print(f"  {g:18s} {m['norm2'] / T:10.4f} {m['margin']:+9.2f} {100 * m['margin'] / Tm:8.1f}%"
                  f"  {m['sum_individual_norm2'] / T:14.4f}")

    summary = {"experiment": cfg["experiment"], "hypothesis": None, "exploratory": True,
               "git_commit": git_commit(), "runs": results, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
