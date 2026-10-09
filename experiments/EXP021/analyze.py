"""
EXP021: H023 (H001 on new data splits) and H024 (is the frequency set fixed by the init seed or by
the split?) on a crossed design: data_seed {598, 101, 202} x model seed {0, 1, 2}, 1-layer model.
Committed before the new-split models were trained.

    uv run python experiments/EXP021/analyze.py --config experiments/EXP021/analysis.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.dynamics import alias_margin, freq_set, jaccard, two_way_decomposition  # noqa: E402
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
    c23, c24 = cfg["H023"], cfg["H024"]

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    splits, seeds = list(cfg["runs"]), cfg["seeds"]
    R = {}
    for sp in splits:
        for sd in seeds:
            rdir = ROOT / cfg["runs"][sp] / f"seed{sd}"
            s = json.loads((rdir / "summary.json").read_text())
            assert str(s["data_seed"]) == sp and s["seed"] == sd, f"{rdir}: wrong data_seed/seed"
            mc = ModelConfig.from_dict(s["config"]["model"])
            assert mc.n_layers == 1
            m = Transformer(mc)
            m.load_state_dict(torch.load(rdir / "checkpoints" / cfg["checkpoint"], map_location="cpu", weights_only=True))
            assert state_sha256(m) == s["final_state_sha256"], f"{rdir}: checkpoint hash mismatch"
            p = s["config"]["task"]["p"]
            ks, sh = freq_set(m.W_E.detach(), p, cfg["major_frequency_min_share"])
            R[(sp, sd)] = {"test_acc": s["final"]["test_acc"], "first_test": s["first_step_test_acc_ge_0.99"],
                           "split_sha256": s["split_sha256"], "init_sha256": s["init_state_sha256"],
                           "freqs": ks, "freq_shares": sh, "alias_rel_margin": alias_margin(ks, p)["rel_margin"]}

    # design checks: same seed -> identical init on every split; different splits -> different split hash
    for sd in seeds:
        assert len({R[(sp, sd)]["init_sha256"] for sp in splits}) == 1, f"seed {sd}: init differs across splits"
    assert len({R[(sp, seeds[0])]["split_sha256"] for sp in splits}) == len(splits), "splits are not distinct"

    # ---- H023 ----
    new = [(sp, sd) for sp in cfg["new_splits"] for sd in seeds]
    passed = [R[k]["test_acc"] >= c23["acc_threshold"] for k in new]
    lo, hi = c23["first_test_band"]
    in_band = [R[k]["first_test"] is not None and lo <= R[k]["first_test"] <= hi for k in new]
    v23 = {"H023 primary (pass >= %d/%d)" % (c23["min_pass"], len(new)): "MET" if sum(passed) >= c23["min_pass"] else "NOT MET",
           "H023 secondary (first_test in band, all)": "MET" if all(in_band) else "NOT MET"}

    rlo, rhi = cfg["reference_first_test_range"]
    split_flag = {}
    for sp in cfg["new_splits"]:
        ft = [R[(sp, sd)]["first_test"] for sd in seeds]
        below = all(x is not None and x < rlo for x in ft)
        above = all(x is None or x > rhi for x in ft)
        split_flag[sp] = "below reference range" if below else ("above reference range" if above else "no flag")

    log_ft = [[math.log10(R[(sp, sd)]["first_test"]) if R[(sp, sd)]["first_test"] else None for sd in seeds] for sp in splits]
    decomp_ft = two_way_decomposition(log_ft) if all(x is not None for row in log_ft for x in row) else None
    decomp_K = two_way_decomposition([[len(R[(sp, sd)]["freqs"]) for sd in seeds] for sp in splits])

    # ---- H024 ----
    ok = lambda k: R[k]["test_acc"] >= c24["model_min_acc"]
    same = [(sd, a, b, jaccard(R[(a, sd)]["freqs"], R[(b, sd)]["freqs"]))
            for sd in seeds for a, b in itertools.combinations(splits, 2) if ok((a, sd)) and ok((b, sd))]
    cross = [(sp, x, y, jaccard(R[(sp, x)]["freqs"], R[(sp, y)]["freqs"]))
             for sp in splits for x, y in itertools.combinations(seeds, 2) if ok((sp, x)) and ok((sp, y))]
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    m_same, m_cross = mean([j for *_, j in same]), mean([j for *_, j in cross])
    if len(same) < c24["min_valid_pairs"]:
        v24 = "UNDECIDABLE"
    elif not (m_cross <= c24["control_max_cross_seed_jaccard"]):
        v24 = "INVALID"
    elif m_same >= c24["met_min_mean_jaccard"]:
        v24 = "MET"
    elif m_same <= c24["not_met_max_mean_jaccard"]:
        v24 = "NOT MET"
    else:
        v24 = "UNDECIDABLE"
    verdicts = {**v23, "H024 (same-seed Jaccard)": v24}

    print("=== EXP021: crossed design data_seed x model seed (1-layer) ===")
    print(f"{'split':>5s} {'seed':>4s} {'test_acc':>8s} {'first_test':>10s}  freqs (alias rel_margin)")
    for (sp, sd), r in R.items():
        print(f"{sp:>5s} {sd:4d} {r['test_acc']:8.4f} {str(r['first_test']):>10s}  {r['freqs']} ({r['alias_rel_margin']:.3f})")
    print(f"\nH023: passed {sum(passed)}/{len(new)} new runs; in band {sum(in_band)}/{len(new)}; split flags {split_flag}")
    if decomp_ft:
        print("variance shares of log10(first_test) [descriptive, df tiny]: " +
              ", ".join(f"{k} {v:.2f}" for k, v in decomp_ft["share"].items()) + "  (rows = split, cols = seed)")
    else:
        print("variance decomposition of log10(first_test): SKIPPED (a run never reached 0.99)")
    print("variance shares of K: " + ", ".join(f"{k} {v:.2f}" for k, v in decomp_K["share"].items()))
    print(f"H024: mean same-seed Jaccard {m_same:.3f} over {len(same)} pairs | "
          f"control: mean cross-seed Jaccard {m_cross:.3f} over {len(cross)} pairs")
    for sd, a, b, j in same:
        print(f"    seed {sd}: {a} vs {b}  J = {j:.2f}")
    print("\n=== PREDICTIONS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(), "verdicts": verdicts,
               "runs": {f"{sp}/seed{sd}": r for (sp, sd), r in R.items()},
               "split_flags": split_flag, "decomposition_log10_first_test": decomp_ft, "decomposition_K": decomp_K,
               "jaccard_same_seed": same, "jaccard_cross_seed": cross,
               "mean_same": m_same, "mean_cross": m_cross, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
