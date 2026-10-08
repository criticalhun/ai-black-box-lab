"""
EXP008: confirmatory tests of H004–H007 on FRESH models (EXP005, seeds 3–5).
Criteria were preregistered in hypotheses/H004–H007.md (commit 47c4f4d).

    uv run python experiments/EXP008/confirm.py --config experiments/EXP008/config.yaml
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
from src.interventions import mlp_class_ablation_hook  # noqa: E402
from src.mech import attention_from_eq, logit_fit, major_frequencies, neuron_structure  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402

HOOK = "blocks.0.mlp.hook_post"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def verdict(passes: list[bool], invalid: bool) -> str:
    if invalid:
        return "INVALID"
    n = sum(passes)
    return "SUPPORTED" if n == len(passes) else ("PARTIAL" if n >= 1 else "REJECTED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    src, ev = cfg["source"], cfg["evaluation"]
    c4, c5, c6, c7 = cfg["H004"], cfg["H005"], cfg["H006"], cfg["H007"]
    assert ev == {"inputs": "all_pairs", "acc_split": "test", "device": "cpu"}

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
        return {"acc": (lg.argmax(-1) == y).double().mean().item(),
                "loss": F.cross_entropy(lg, y).item()}

    res = {}
    for run in src["runs"]:
        rdir = exp / "results" / run
        s = json.loads((rdir / "summary.json").read_text())
        final = load(rdir, src["checkpoint"], s["final_state_sha256"])
        init = load(rdir, src["control_checkpoint"], s["init_state_sha256"])
        major, major_share = major_frequencies(final, p, cfg["major_frequency_min_share"])
        with torch.no_grad():
            lf, cf = final.run_with_cache(tokens)
            li, ci = init.run_with_cache(tokens)

        # H004
        att = attention_from_eq(cf)
        h4 = all(c4["to_a_range"][0] <= h["to_a"] <= c4["to_a_range"][1]
                 and c4["to_b_range"][0] <= h["to_b"] <= c4["to_b_range"][1]
                 and h["to_eq"] <= c4["to_eq_max"] for h in att)
        # H005
        nf, ni = neuron_structure(cf, p, major), neuron_structure(ci, p, major)
        h5 = (bool(major)
              and nf["frac_alive_dominant_in_major"] >= c5["min_frac_alive_dominant_in_major"]
              and nf["class_shares"]["same_freq"] >= c5["min_same_freq_share"])
        h5_ctrl_ok = ni["class_shares"]["same_freq"] < c5["control_max_same_freq_share"]
        # H006
        fit_f, fit_i = logit_fit(lf, p, major), logit_fit(li, p, major)
        h6 = (bool(major) and fit_f["r2"] >= c6["min_r2"]
              and all(v > 0 for v in fit_f["coef"].values()))
        h6_ctrl_ok = (not major) or fit_i["r2"] < c6["control_max_r2"]
        # H007
        abl = {}
        with torch.no_grad():
            for name, remove in [("baseline", set()), ("cross", {"cross"}),
                                 ("same_freq", {"same_freq"}), ("univariate", {"a_only", "b_only"})]:
                lg = final.run_with_hooks(tokens, [(HOOK, mlp_class_ablation_hook(p, remove))])
                abl[name] = test_metrics(lg)
        h7_ctrl_ok = (abl["baseline"]["acc"] >= c7["baseline_min_acc"]
                      and abl["cross"]["acc"] >= c7["control_cross_min_acc"])
        h7 = (abl["same_freq"]["acc"] <= c7["same_freq_removed_max_acc"]
              and abl["univariate"]["acc"] >= c7["univariate_removed_min_acc"])

        res[run] = {"major_freqs": major, "major_share": major_share,
                    "H004": {"pass": h4, "heads": att},
                    "H005": {"pass": h5, "control_ok": h5_ctrl_ok, "final": nf, "init": ni},
                    "H006": {"pass": h6, "control_ok": h6_ctrl_ok, "final": fit_f, "init_r2": fit_i["r2"]},
                    "H007": {"pass": h7, "control_ok": h7_ctrl_ok, "ablations": abl}}

    runs = list(res.values())
    verdicts = {
        "H004": verdict([r["H004"]["pass"] for r in runs], False),
        "H005": verdict([r["H005"]["pass"] for r in runs], not all(r["H005"]["control_ok"] for r in runs)),
        "H006": verdict([r["H006"]["pass"] for r in runs], not all(r["H006"]["control_ok"] for r in runs)),
        "H007": verdict([r["H007"]["pass"] for r in runs], not all(r["H007"]["control_ok"] for r in runs)),
    }

    print("=== EXP008: confirmatory tests on FRESH seeds (preregistered: commit 47c4f4d) ===")
    for run, r in res.items():
        print(f"\n##### {run}  major freqs (W_E share >= {cfg['major_frequency_min_share']}): "
              + ", ".join(f"{k} ({v:.3f})" for k, v in zip(r["major_freqs"], r["major_share"])))
        print(f"H004 {'PASS' if r['H004']['pass'] else 'FAIL'} | attention from '=': " + "; ".join(
            f"h{h['head']} a {h['to_a']:.3f} b {h['to_b']:.3f} = {h['to_eq']:.3f}" for h in r["H004"]["heads"]))
        h5 = r["H005"]
        print(f"H005 {'PASS' if h5['pass'] else 'FAIL'} | alive {h5['final']['n_alive']}, dominant in major "
              f"{h5['final']['frac_alive_dominant_in_major']:.3f} | same_freq {h5['final']['class_shares']['same_freq']:.3f}"
              f" (init {h5['init']['class_shares']['same_freq']:.3f}, control {'ok' if h5['control_ok'] else 'FAIL'})")
        h6 = r["H006"]
        print(f"H006 {'PASS' if h6['pass'] else 'FAIL'} | R² {h6['final']['r2']:.4f} (init {h6['init_r2']:.4f}, "
              f"control {'ok' if h6['control_ok'] else 'FAIL'}) | α: "
              + ", ".join(f"{k}: {v:+.2f}" for k, v in h6["final"]["coef"].items()))
        a = r["H007"]["ablations"]
        print(f"H007 {'PASS' if r['H007']['pass'] else 'FAIL'} | test acc (loss): "
              + " | ".join(f"{n} {m['acc']:.4f} ({m['loss']:.2e})" for n, m in a.items())
              + f" | control {'ok' if r['H007']['control_ok'] else 'FAIL'}")

    print("\n=== VERDICTS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(),
               "verdicts": verdicts, "runs": res, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
