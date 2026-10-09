"""
EXP016 (EXPLORATORY): in the 2-layer model, which layer / position does the computation?
No hypothesis, no verdict.

    uv run python experiments/EXP016/explore.py --config experiments/EXP016/config.yaml
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
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from src.data import modular_addition  # noqa: E402
from src.interventions import mean_ablation_hook, mlp_class_ablation_hook  # noqa: E402
from src.mech import major_frequencies, neuron_structure_at, residual_dla  # noqa: E402
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
    sites = [tuple(s) for s in cfg["mlp_sites"]]
    assert cfg["evaluation"] == {"inputs": "all_pairs", "acc_split": "test", "device": "cpu"}

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    exp = ROOT / "experiments" / src["experiment"]
    exp_cfg = yaml.safe_load((exp / "config.yaml").read_text())
    model_cfg = ModelConfig.from_dict(exp_cfg["model"])
    task = exp_cfg["task"]
    p = task["p"]
    ds = modular_addition(p, task["train_fraction"], task["data_seed"])
    tokens, labels, test_idx = ds.tokens, ds.labels, ds.test_idx

    def load(rdir: Path, name: str, expected: str) -> Transformer:
        m = Transformer(model_cfg)
        m.load_state_dict(torch.load(rdir / "checkpoints" / name, map_location="cpu", weights_only=True))
        assert state_sha256(m) == expected, f"{rdir.name}/{name}: checkpoint hash mismatch"
        return m

    def test_metrics(logits: torch.Tensor) -> dict:
        lg = logits[test_idx, -1, :].to(torch.float64)
        y = labels[test_idx]
        return {"acc": (lg.argmax(-1) == y).double().mean().item(), "loss": F.cross_entropy(lg, y).item()}

    n_layers = model_cfg.n_layers
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

        structure = {f"L{l}p{q}": {"final": neuron_structure_at(cf, p, major, l, q),
                                   "init": neuron_structure_at(ci, p, major, l, q)} for l, q in sites}
        dla = residual_dla(final, tokens, p)

        abl = {}
        with torch.no_grad():
            abl["baseline"] = test_metrics(final(tokens))
            for l in range(n_layers):
                for comp in ("attn", "mlp"):
                    hk = f"blocks.{l}.hook_{comp}_out"
                    abl[f"mean-ablate {comp}{l}"] = test_metrics(final.run_with_hooks(tokens, [(hk, mean_ablation_hook())]))
            for l, q in sites:
                hk = f"blocks.{l}.mlp.hook_post"
                for name, remove in (("same_freq", {"same_freq"}), ("univariate", {"a_only", "b_only"})):
                    lg = final.run_with_hooks(tokens, [(hk, mlp_class_ablation_hook(p, remove, pos=q))])
                    abl[f"L{l}p{q} remove {name}"] = test_metrics(lg)

        results[run] = {"major_freqs": major, "major_share": major_share,
                        "structure": structure, "dla": dla, "ablations": abl}

    print("=== EXP016 — EXPLORATORY: where does the 2-layer model compute? (no verdict) ===")
    print("pos 1 = b's position (already sees a), pos 2 = '=' (where the answer is read)")
    for run, r in results.items():
        print(f"\n##### {run}   major freqs: " + ", ".join(
            f"{k} ({v:.2f})" for k, v in zip(r["major_freqs"], r["major_share"])))
        print("  MLP neuron structure         alive  dom.in major  same_freq  (init same_freq)")
        for site, st in r["structure"].items():
            f_, i_ = st["final"], st["init"]
            print(f"    {site:6s}                     {f_['n_alive']:4d}   {f_['frac_alive_dominant_in_major']:10.3f}"
                  f"  {f_['class_shares']['same_freq']:9.3f}  ({i_['class_shares']['same_freq']:.3f})")
        tm = r["dla"]["total"]["margin"]
        print(f"  direct logit attribution (margin share of {tm:+.2f}): " + ", ".join(
            f"{k} {100 * v['margin'] / tm:5.1f}%" for k, v in r["dla"].items() if k != "total"))
        print("  interventions (test acc, loss):")
        for name, m in r["ablations"].items():
            print(f"    {name:28s} acc {m['acc']:.4f}  loss {m['loss']:.2e}")

    summary = {"experiment": cfg["experiment"], "hypothesis": None, "exploratory": True,
               "git_commit": git_commit(), "runs": results, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
