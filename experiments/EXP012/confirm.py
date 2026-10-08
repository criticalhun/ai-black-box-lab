"""
EXP012: confirmatory tests of H008–H009 on FRESH models (EXP011, seeds 6–8).
Criteria preregistered in hypotheses/H008–H009.md before these models were trained.

    uv run python experiments/EXP012/confirm.py --config experiments/EXP012/config.yaml
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
from src.mech import dla_metrics, logit_components, mlp_class_coherence  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402

UNIVARIATE = ["attn:a_only", "attn:b_only", "mlp:a_only", "mlp:b_only"]


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
    src = cfg["source"]
    c8, c9 = cfg["H008"], cfg["H009"]
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

    lo, hi = c8["control_kappa_range"]
    res = {}
    for run in src["runs"]:
        rdir = exp / "results" / run
        s = json.loads((rdir / "summary.json").read_text())
        final = load(rdir, src["checkpoint"], s["final_state_sha256"])
        init = load(rdir, src["control_checkpoint"], s["init_state_sha256"])

        # H008 — coherence of MLP neurons' logit contributions
        kf, ki = mlp_class_coherence(final, tokens, p), mlp_class_coherence(init, tokens, p)
        h8 = (kf["univariate"]["kappa"] <= c8["univariate_max_kappa"]
              and kf["same_freq"]["kappa"] >= c8["same_freq_min_kappa"])
        h8_ctrl = all(lo <= ki[g]["kappa"] <= hi for g in ("univariate", "same_freq"))

        # H009 — direct logit attribution
        comps, total = logit_components(final, tokens, p)
        tot = dla_metrics(total, p)
        sf_margin = dla_metrics(comps["mlp:same_freq"], p)["margin"]
        uni_norm = dla_metrics(sum(comps[k] for k in UNIVARIATE), p)["norm2"]
        margin_share = sf_margin / tot["margin"] if tot["margin"] > 0 else float("nan")
        uni_share = uni_norm / tot["norm2"]
        h9 = (tot["margin"] >= c9["min_total_margin"] and margin_share >= c9["mlp_same_freq_min_margin_share"]
              and uni_share <= c9["univariate_max_norm_share"])

        res[run] = {
            "H008": {"pass": h8, "control_ok": h8_ctrl,
                     "final": {g: kf[g]["kappa"] for g in kf}, "init": {g: ki[g]["kappa"] for g in ki}},
            "H009": {"pass": h9, "total_margin": tot["margin"],
                     "mlp_same_freq_margin_share": margin_share, "univariate_norm_share": uni_share},
        }

    runs = list(res.values())
    verdicts = {
        "H008": verdict([r["H008"]["pass"] for r in runs], not all(r["H008"]["control_ok"] for r in runs)),
        "H009": verdict([r["H009"]["pass"] for r in runs], False),
    }

    print("=== EXP012: confirmatory tests on FRESH seeds 6–8 ===")
    for run, r in res.items():
        a, b = r["H008"], r["H009"]
        print(f"\n##### {run}")
        print(f"H008 {'PASS' if a['pass'] else 'FAIL'} | kappa final: univariate {a['final']['univariate']:.4f}, "
              f"same_freq {a['final']['same_freq']:.2f}, cross {a['final']['cross']:.2f} | init: univariate "
              f"{a['init']['univariate']:.4f}, same_freq {a['init']['same_freq']:.4f} "
              f"(control {'ok' if a['control_ok'] else 'FAIL'})")
        print(f"H009 {'PASS' if b['pass'] else 'FAIL'} | total margin {b['total_margin']:+.2f} | "
              f"mlp:same_freq margin share {b['mlp_same_freq_margin_share']:.4f} | "
              f"univariate norm share {b['univariate_norm_share']:.4f}")

    print("\n=== VERDICTS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(),
               "verdicts": verdicts, "runs": res, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
