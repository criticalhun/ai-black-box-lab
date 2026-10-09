"""
EXP018: confirmatory tests of H014–H017 on FRESH 2-layer models (EXP017, seeds 3–5).
Criteria and this code were committed before these models were trained.

    uv run python experiments/EXP018/confirm.py --config experiments/EXP018/config.yaml
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
    c14, c15, c16, c17 = cfg["H014"], cfg["H015"], cfg["H016"], cfg["H017"]
    assert cfg["evaluation"] == {"inputs": "all_pairs", "acc_split": "test", "device": "cpu"}

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    exp = ROOT / "experiments" / src["experiment"]
    exp_cfg = yaml.safe_load((exp / "config.yaml").read_text())
    model_cfg = ModelConfig.from_dict(exp_cfg["model"])
    assert model_cfg.n_layers == 2, "EXP018 is defined for 2-layer models"
    task = exp_cfg["task"]
    p = task["p"]
    ds = modular_addition(p, task["train_fraction"], task["data_seed"])
    tokens, labels, test_idx = ds.tokens, ds.labels, ds.test_idx

    def load(rdir: Path, name: str, expected: str) -> Transformer:
        m = Transformer(model_cfg)
        m.load_state_dict(torch.load(rdir / "checkpoints" / name, map_location="cpu", weights_only=True))
        assert state_sha256(m) == expected, f"{rdir.name}/{name}: checkpoint hash mismatch"
        return m

    def acc(logits: torch.Tensor) -> float:
        lg = logits[test_idx, -1, :].to(torch.float64)
        return (lg.argmax(-1) == labels[test_idx]).double().mean().item()

    res = {}
    for run in src["runs"]:
        rdir = exp / "results" / run
        s = json.loads((rdir / "summary.json").read_text())
        final = load(rdir, src["checkpoint"], s["final_state_sha256"])
        init = load(rdir, src["control_checkpoint"], s["init_state_sha256"])
        major, major_share = major_frequencies(final, p, cfg["major_frequency_min_share"])
        with torch.no_grad():
            _, cf = final.run_with_cache(tokens)
            _, ci = init.run_with_cache(tokens)
            base = acc(final(tokens))
        valid = base >= cfg["baseline_min_acc"]

        # H014 — structure
        sf = lambda c, l, q: neuron_structure_at(c, p, major, l, q)["class_shares"]["same_freq"]
        l1, l0, l1_init = sf(cf, 1, 2), sf(cf, 0, 2), sf(ci, 1, 2)
        h14 = l1 >= c14["L1p2_min_same_freq"] and l0 <= c14["L0p2_max_same_freq"]
        h14_ctrl = l1_init < c14["control_L1p2_max_same_freq"]

        # H015 — direct logit attribution
        d = residual_dla(final, tokens, p)
        tm = d["total"]["margin"]
        mlp1_share = d["mlp1"]["margin"] / tm if tm > 0 else float("nan")
        l0_share = (d["attn0"]["margin"] + d["mlp0"]["margin"]) / tm if tm > 0 else float("nan")
        h15 = tm >= c15["min_total_margin"] and mlp1_share >= c15["mlp1_min_margin_share"] and abs(l0_share) <= c15["layer0_max_margin_share"]

        # H016 — causal division of labour; H017 — every component needed
        with torch.no_grad():
            run_h = lambda name, hook: acc(final.run_with_hooks(tokens, [(name, hook)]))
            a_l1_sf = run_h("blocks.1.mlp.hook_post", mlp_class_ablation_hook(p, {"same_freq"}, pos=2))
            a_l1_uni = run_h("blocks.1.mlp.hook_post", mlp_class_ablation_hook(p, {"a_only", "b_only"}, pos=2))
            a_l0_uni = run_h("blocks.0.mlp.hook_post", mlp_class_ablation_hook(p, {"a_only", "b_only"}, pos=2))
            mean_abl = {f"{c}{l}": run_h(f"blocks.{l}.hook_{c}_out", mean_ablation_hook())
                        for l in (0, 1) for c in ("attn", "mlp")}
        h16 = (a_l1_sf <= c16["L1p2_remove_same_freq_max_acc"]
               and a_l1_uni >= c16["L1p2_remove_univariate_min_acc"]
               and a_l0_uni <= c16["L0p2_remove_univariate_max_acc"])
        h17 = all(v <= c17["mean_ablate_each_max_acc"] for v in mean_abl.values())

        res[run] = {"baseline_acc": base, "valid": valid, "major_freqs": major, "major_share": major_share,
                    "H014": {"pass": h14, "control_ok": h14_ctrl, "L1p2_same_freq": l1,
                             "L0p2_same_freq": l0, "L1p2_same_freq_init": l1_init},
                    "H015": {"pass": h15, "total_margin": tm, "mlp1_share": mlp1_share, "layer0_share": l0_share},
                    "H016": {"pass": h16, "L1p2_remove_same_freq": a_l1_sf,
                             "L1p2_remove_univariate": a_l1_uni, "L0p2_remove_univariate": a_l0_uni},
                    "H017": {"pass": h17, "mean_ablation_acc": mean_abl}}

    runs = list(res.values())
    invalid = not all(r["valid"] for r in runs)
    verdicts = {
        "H014": verdict([r["H014"]["pass"] for r in runs], invalid or not all(r["H014"]["control_ok"] for r in runs)),
        "H015": verdict([r["H015"]["pass"] for r in runs], invalid),
        "H016": verdict([r["H016"]["pass"] for r in runs], invalid),
        "H017": verdict([r["H017"]["pass"] for r in runs], invalid),
    }

    print("=== EXP018: confirmatory tests on FRESH 2-layer models (seeds 3–5) ===")
    ok = lambda b: "PASS" if b else "FAIL"
    for run, r in res.items():
        a, b, c, e = r["H014"], r["H015"], r["H016"], r["H017"]
        print(f"\n##### {run}  baseline acc {r['baseline_acc']:.4f}  major freqs: " + ", ".join(
            f"{k} ({v:.2f})" for k, v in zip(r["major_freqs"], r["major_share"])))
        print(f"H014 {ok(a['pass'])} | same_freq L1p2 {a['L1p2_same_freq']:.3f} (init {a['L1p2_same_freq_init']:.3f}), "
              f"L0p2 {a['L0p2_same_freq']:.3f}")
        print(f"H015 {ok(b['pass'])} | total margin {b['total_margin']:+.2f} | mlp1 share {b['mlp1_share']:.3f} | "
              f"layer0 share {b['layer0_share']:+.3f}")
        print(f"H016 {ok(c['pass'])} | L1p2 remove same_freq acc {c['L1p2_remove_same_freq']:.4f} | "
              f"L1p2 remove univariate acc {c['L1p2_remove_univariate']:.4f} | "
              f"L0p2 remove univariate acc {c['L0p2_remove_univariate']:.4f}")
        print(f"H017 {ok(e['pass'])} | mean-ablation acc: " + ", ".join(
            f"{k} {v:.4f}" for k, v in e["mean_ablation_acc"].items()))

    print("\n=== VERDICTS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(),
               "verdicts": verdicts, "runs": res, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
