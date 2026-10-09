"""
EXP020: confirmatory tests of H018–H022 (what flows between the layers of the 2-layer model)
on EXP017 models (seeds 3–5), whose between-layer quantities were never inspected before.

    uv run python experiments/EXP020/confirm.py --config experiments/EXP020/config.yaml
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
from src.mech import class_composition, major_frequencies, univariate_spectrum  # noqa: E402
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
    c18, c19, c20, c21, c22 = (cfg[h] for h in ("H018", "H019", "H020", "H021", "H022"))
    assert cfg["evaluation"] == {"inputs": "all_pairs", "acc_split": "test", "device": "cpu"}

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
    ds = modular_addition(p, task["train_fraction"], task["data_seed"])
    tokens, labels, test_idx = ds.tokens, ds.labels, ds.test_idx

    def load(rdir: Path, name: str, expected: str) -> Transformer:
        m = Transformer(model_cfg)
        m.load_state_dict(torch.load(rdir / "checkpoints" / name, map_location="cpu", weights_only=True))
        assert state_sha256(m) == expected, f"{rdir.name}/{name}: checkpoint hash mismatch"
        return m

    grid = lambda cache, hook: cache[hook][:, 2].to(torch.float64).reshape(p, p, -1)   # "=" position

    res = {}
    for run in src["runs"]:
        rdir = exp / "results" / run
        s = json.loads((rdir / "summary.json").read_text())
        final = load(rdir, src["checkpoint"], s["final_state_sha256"])
        init = load(rdir, src["control_checkpoint"], s["init_state_sha256"])
        major, major_share = major_frequencies(final, p, cfg["major_frequency_min_share"])
        with torch.no_grad():
            logits, cf = final.run_with_cache(tokens)
            _, ci = init.run_with_cache(tokens)
        base = (logits[test_idx, -1].argmax(-1) == labels[test_idx]).double().mean().item()

        r0 = class_composition(grid(cf, "blocks.0.hook_resid_post"), major)
        dom = "a_only" if r0["a_only_share"] >= r0["b_only_share"] else "b_only"
        other = "b_only" if dom == "a_only" else "a_only"
        dom_share = r0[f"{dom}_share"]
        sp = univariate_spectrum(grid(cf, "blocks.0.hook_mlp_out"), major, p)
        a1 = class_composition(grid(cf, "blocks.1.hook_attn_out"), major)
        m1f = class_composition(grid(cf, "blocks.1.hook_mlp_out"), major)
        m1i = class_composition(grid(ci, "blocks.1.hook_mlp_out"), major)

        res[run] = {
            "baseline_acc": base, "valid": base >= cfg["baseline_min_acc"],
            "major_freqs": major, "major_share": major_share,
            "H018": {"pass": dom_share >= c18["min_dominant_share"], "dominant": dom, "dominant_share": dom_share,
                     "resid_post0": r0},
            "H019": {"pass": dom == c19["dominant"] and dom_share >= c18["min_dominant_share"], "dominant": dom},
            "H020": {"pass": sp["major"] >= c20["min_major_share"] and sp["harmonics"] <= c20["max_harmonics_share"],
                     "spectrum": sp},
            "H021": {"pass": a1[f"{other}_share"] >= c21["min_other_share"], "other": other,
                     "other_share": a1[f"{other}_share"], "attn1_out": a1},
            "H022": {"pass": m1f["same_freq_share"] >= c22["min_same_freq"],
                     "control_ok": m1i["same_freq_share"] < c22["control_max_same_freq"],
                     "same_freq": m1f["same_freq_share"], "same_freq_init": m1i["same_freq_share"]},
        }

    runs = list(res.values())
    invalid = not all(r["valid"] for r in runs)
    verdicts = {h: verdict([r[h]["pass"] for r in runs], invalid) for h in ("H018", "H019", "H020", "H021")}
    verdicts["H022"] = verdict([r["H022"]["pass"] for r in runs],
                               invalid or not all(r["H022"]["control_ok"] for r in runs))

    print("=== EXP020: what flows between the layers — confirmatory (EXP017 seeds 3–5) ===")
    ok = lambda b: "PASS" if b else "FAIL"
    for run, r in res.items():
        print(f"\n##### {run}  baseline acc {r['baseline_acc']:.4f}  major freqs: " + ", ".join(
            f"{k} ({v:.2f})" for k, v in zip(r["major_freqs"], r["major_share"])))
        a, b, c, d, e = (r[h] for h in ("H018", "H019", "H020", "H021", "H022"))
        print(f"H018 {ok(a['pass'])} | after L0 @'=': dominant {a['dominant']} {a['dominant_share']:.3f} "
              f"(a_only {a['resid_post0']['a_only_share']:.3f}, b_only {a['resid_post0']['b_only_share']:.3f})")
        print(f"H019 {ok(b['pass'])} | dominant number = {b['dominant']}")
        print(f"H020 {ok(c['pass'])} | MLP0 out univariate energy: major {c['spectrum']['major']:.3f}, "
              f"harmonics {c['spectrum']['harmonics']:.3f}, other {c['spectrum']['other']:.3f}")
        print(f"H021 {ok(d['pass'])} | attn1 out @'=': other number ({d['other']}) share {d['other_share']:.3f}")
        print(f"H022 {ok(e['pass'])} | MLP1 out @'=' same_freq {e['same_freq']:.3f} (init {e['same_freq_init']:.3f})")

    print("\n=== VERDICTS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(),
               "verdicts": verdicts, "runs": res, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
