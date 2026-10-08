"""
EXP003: causal ablation of key Fourier frequencies in W_E (tests H003).

    uv run python experiments/EXP003/run.py --config experiments/EXP003/config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data import modular_addition  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.fourier import frequency_energy  # noqa: E402
from src.interventions import embed_keep_frequencies  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    src, iv, ev = cfg["source"], cfg["intervention"], cfg["evaluation"]
    cr, ct, ex = cfg["criteria"], cfg["control"], cfg["exploratory"]
    assert iv["matrix"] == "W_E" and iv["basis"] == "real_fourier_orthonormal"
    assert ev["split"] == "test" and ev["device"] == "cpu"

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    exp1 = ROOT / "experiments" / src["experiment"]
    exp1_cfg = yaml.safe_load((exp1 / "config.yaml").read_text())
    model_cfg = ModelConfig.from_dict(exp1_cfg["model"])
    task = exp1_cfg["task"]
    p = task["p"]
    rows = tuple(iv["rows"])
    assert rows == (0, p)
    x_te, y_te = modular_addition(p, task["train_fraction"], task["data_seed"]).test
    exp2 = json.loads((ROOT / "experiments" / cfg["key_frequencies_from"]
                       / "results" / "summary.json").read_text())

    n_freq = (p - 1) // 2
    ALL = set(range(n_freq + 1))
    NONCONST = set(range(1, n_freq + 1))

    def ev_keep(model, keep):
        return evaluate(embed_keep_frequencies(model, keep, rows), x_te, y_te)

    results = {}
    for idx, run in enumerate(src["runs"]):
        rdir = exp1 / "results" / run
        s1 = json.loads((rdir / "summary.json").read_text())
        model = Transformer(model_cfg)
        model.load_state_dict(torch.load(rdir / "checkpoints" / src["checkpoint"],
                                         map_location="cpu", weights_only=True))
        assert state_sha256(model) == s1["final_state_sha256"], f"{run}: checkpoint hash mismatch"
        key = set(exp2["runs"][run]["final"]["top_k_freqs"])

        # ---- preregistered ----
        baseline = evaluate(model, x_te, y_te)
        necessity = ev_keep(model, ALL - key)
        keep_suff = key | ({0} if iv["keep_constant_in_sufficiency"] else set())
        sufficiency = ev_keep(model, keep_suff)
        rng = random.Random(ct["rng_seed"] + 1000 * idx)
        non_key = sorted(NONCONST - key)
        controls = []
        for _ in range(ct["n_draws"]):
            drop = sorted(rng.sample(non_key, ct["n_freqs"]))
            controls.append({"dropped": drop, **ev_keep(model, ALL - set(drop))})
        passed = (necessity["acc"] <= cr["necessity_max_acc"]
                  and sufficiency["acc"] >= cr["sufficiency_min_acc"])

        # ---- exploratory ----
        E = frequency_energy(model.W_E[rows[0]:rows[1]])
        frac = E / E.sum()
        key_sorted = sorted(key, key=lambda k: -frac[k].item())
        dose, single = [], []
        if ex["dose_response"]:
            for i in range(1, len(key_sorted) + 1):
                removed = key_sorted[:i]
                dose.append({"removed": removed, **ev_keep(model, ALL - set(removed))})
        if ex["single_frequency_ablation"]:
            for k in key_sorted:
                single.append({"freq": k, "energy_share": frac[k].item(), **ev_keep(model, ALL - {k})})

        results[run] = {"key_freqs": sorted(key), "baseline": baseline, "necessity": necessity,
                        "sufficiency": sufficiency, "controls": controls, "pass": passed,
                        "exploratory": {"key_by_energy": key_sorted, "dose_response": dose,
                                        "single_ablation": single}}

    invalid = any(r["baseline"]["acc"] < cr["baseline_min_acc"]
                  or any(c["acc"] < cr["control_min_acc"] for c in r["controls"])
                  for r in results.values())
    n_pass = sum(r["pass"] for r in results.values())
    if invalid:
        verdict = "INVALID"
    elif n_pass == len(results):
        verdict = "SUPPORTED"
    elif n_pass >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"

    print("=== H003: preregistered test (test split, CPU) ===")
    print(f"criteria: baseline>={cr['baseline_min_acc']}, necessity<={cr['necessity_max_acc']}, "
          f"sufficiency>={cr['sufficiency_min_acc']}, every control>={cr['control_min_acc']}")
    for run, r in results.items():
        cmin = min(c["acc"] for c in r["controls"])
        print(f"{run:6s} baseline {r['baseline']['acc']:.4f} | necessity {r['necessity']['acc']:.4f} "
              f"(loss {r['necessity']['loss']:.2e}) | sufficiency {r['sufficiency']['acc']:.4f} "
              f"(loss {r['sufficiency']['loss']:.2e}) | controls min {cmin:.4f} | "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    print(f"VERDICT: {verdict} ({n_pass}/{len(results)} runs pass)")

    print("\n=== EXPLORATORY (not part of verdict) ===")
    for run, r in results.items():
        e = r["exploratory"]
        print(f"\n{run}: key freqs by energy share: "
              + ", ".join(f"{s['freq']} ({s['energy_share']:.3f})" for s in e["single_ablation"]))
        print("  dose-response (remove top-i by energy):")
        for i, d in enumerate(e["dose_response"], 1):
            print(f"    i={i}  acc {d['acc']:.4f}  loss {d['loss']:.2e}")
        print("  single-frequency removal:")
        for s in e["single_ablation"]:
            print(f"    k={s['freq']:2d}  acc {s['acc']:.4f}  loss {s['loss']:.2e}")

    summary = {"experiment": cfg["experiment"], "hypothesis": cfg["hypothesis"],
               "git_commit": git_commit(), "verdict": verdict, "n_pass": n_pass,
               "runs": results, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
