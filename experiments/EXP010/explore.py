"""
EXP010 (EXPLORATORY): do MLP neurons' univariate logit contributions cancel each other
while their same-frequency (product) contributions reinforce? No hypothesis, no verdict.

    uv run python experiments/EXP010/explore.py --config experiments/EXP010/config.yaml
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
from src.mech import mlp_class_coherence  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402


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
            "final": mlp_class_coherence(load(rdir, src["checkpoint"], s["final_state_sha256"]), tokens, p),
            "control": mlp_class_coherence(load(rdir, src["control_checkpoint"], s["init_state_sha256"]), tokens, p),
        }

    print("=== EXP010 — EXPLORATORY: coherence of MLP neurons' direct logit contributions ===")
    print("kappa = ||Σ_neurons contribution||² / Σ_neurons ||contribution||²")
    print("        ≈1 incoherent | <<1 neurons cancel each other | >1 neurons reinforce each other")
    print("indiv share = this class's Σ_neurons ||contribution||² / all classes' (before any cancellation)")
    for run, r in results.items():
        f, c = r["final"], r["control"]
        tot_f = sum(f[k]["individual"] for k in ("a_only", "b_only", "same_freq", "cross"))
        tot_c = sum(c[k]["individual"] for k in ("a_only", "b_only", "same_freq", "cross"))
        print(f"\n##### {run}")
        print(f"  {'group':11s} {'kappa final':>11s} {'kappa init':>10s} {'indiv share final':>17s} {'indiv share init':>16s}")
        for g in f:
            print(f"  {g:11s} {f[g]['kappa']:11.4f} {c[g]['kappa']:10.4f} "
                  f"{f[g]['individual'] / tot_f:17.4f} {c[g]['individual'] / tot_c:16.4f}")

    summary = {"experiment": cfg["experiment"], "hypothesis": None, "exploratory": True,
               "git_commit": git_commit(), "runs": results, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
