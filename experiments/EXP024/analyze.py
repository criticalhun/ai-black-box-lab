"""
EXP024: H028 — is the training instability of the 2-layer model caused by the hyperparameters?
Grid lr {3e-4, 1e-3, 3e-3} x weight decay {0, 1.0}, seeds 3–5 (same inits as EXP017).
Committed before the grid was trained.

    uv run python experiments/EXP024/analyze.py --config experiments/EXP024/analysis.yaml
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

from src.dynamics import alias_margin, freq_set, instability, jaccard, read_metrics  # noqa: E402
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
    a, b, c = cfg["H028a"], cfg["H028b"], cfg["H028c"]

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    R, inits = {}, {}
    for cell, (lr, wd, rd) in cfg["cells"].items():
        for sd in cfg["seeds"]:
            rdir = ROOT / rd / f"seed{sd}"
            s = json.loads((rdir / "summary.json").read_text())
            tr = s["config"]["training"]
            assert float(tr["lr"]) == lr and float(tr["weight_decay"]) == wd, f"{rdir}: lr/wd mismatch"
            assert s["data_seed"] == 598 and s["seed"] == sd
            inits.setdefault(sd, set()).add(s["init_state_sha256"])
            mc = ModelConfig.from_dict(s["config"]["model"])
            assert mc.n_layers == 2
            p = s["config"]["task"]["p"]
            m = Transformer(mc)
            m.load_state_dict(torch.load(rdir / "checkpoints" / cfg["checkpoint"], map_location="cpu", weights_only=True))
            assert state_sha256(m) == s["final_state_sha256"], f"{rdir}: checkpoint hash mismatch"
            ks, _ = freq_set(m.W_E.detach(), p, cfg["major_frequency_min_share"])
            ins = instability(read_metrics(rdir / "metrics.csv"), cfg["instability"]["loss_threshold"],
                              cfg["instability"]["buffer_steps"])
            R[(cell, sd)] = {"lr": lr, "wd": wd, "test_acc": s["final"]["test_acc"],
                             "first_test": s["first_step_test_acc_ge_0.99"], "freqs": ks,
                             "rel_margin": alias_margin(ks, p)["rel_margin"], "instability": ins}
    assert all(len(v) == 1 for v in inits.values()), "same seed must give the same init in every cell"

    base = cfg["baseline_cell"]
    # ---- H028a ----
    per_seed = {}
    for sd in cfg["seeds"]:
        S0 = R[(base, sd)]["instability"]["events"]
        Slo = R[(a["low_cell"], sd)]["instability"]["events"]
        hi = R[(a["high_cell"], sd)]["instability"]
        if S0 is None or S0 < a["baseline_min_events"] or Slo is None:
            per_seed[sd] = None
            continue
        hi_ok = hi["events"] is None or hi["events"] >= S0        # never memorized at high lr -> maximally unstable
        per_seed[sd] = Slo <= a["low_max_ratio"] * S0 and hi_ok
    dec = [v for v in per_seed.values() if v is not None]
    v_a = "UNDECIDABLE" if len(dec) < a["min_decidable_seeds"] else ("MET" if all(dec) else "NOT MET")
    # ---- H028b ----
    gen = [R[(cl, sd)]["test_acc"] >= b["acc_threshold"] for cl in b["cells"] for sd in cfg["seeds"]]
    v_b = "MET" if not any(gen) else "NOT MET"
    # ---- H028c ----
    r = R[(c["cell"], c["seed"])]
    J = jaccard(r["freqs"], c["baseline_freqs"])
    if J >= c["m1_min_jaccard"] and len(r["freqs"]) <= 2:
        v_c = "M1-like (init decides)"
    elif len(r["freqs"]) >= 3 and r["test_acc"] >= cfg["stuck_below_acc"]:
        v_c = "M2-like (dynamics decide)"
    else:
        v_c = "neither (undecided)"
    verdicts = {"H028a (instability is lr-driven at wd 1.0)": v_a,
                "H028b (no generalization at wd 0, lr <= 1e-3)": v_b,
                "H028c (seed3 at low lr)": v_c}

    print("=== EXP024: lr x weight-decay grid, 2-layer, seeds 3–5 ===")
    print(f"{'cell':12s} {'seed':>4s} {'test_acc':>8s} {'first':>6s} {'spikes':>6s} {'max loss':>9s}  freqs (m) [J vs base]")
    for (cell, sd), x in R.items():
        ins = x["instability"]
        J0 = jaccard(x["freqs"], R[(base, sd)]["freqs"])
        print(f"{cell:12s} {sd:4d} {x['test_acc']:8.4f} {str(x['first_test']):>6s} {str(ins['events']):>6s} "
              f"{(ins['max_train_loss'] if ins['max_train_loss'] is not None else float('nan')):9.1e}  "
              f"{x['freqs']} ({x['rel_margin']:.3f}) [{J0:.2f}]")
    print(f"\nH028a per seed (None = undecidable): {per_seed}")
    print(f"H028c: seed{c['seed']} @ {c['cell']}: freqs {r['freqs']}, acc {r['test_acc']:.4f}, J vs {c['baseline_freqs']} = {J:.2f}")
    print("\n=== PREDICTIONS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(), "verdicts": verdicts,
               "H028a_per_seed": per_seed, "runs": {f"{k[0]}/seed{k[1]}": v for k, v in R.items()}, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
