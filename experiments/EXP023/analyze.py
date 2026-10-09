"""
EXP023: H026 (alias margin -> residual error) and H027 (number of major frequencies -> residual error)
on FRESH 2-layer models (seeds 9–20, split 598), plus exploratory trajectory measurements that
separate the mechanism candidates M1 (init lottery) and M2 (instability pruning).
Committed before these models were trained.

    uv run python experiments/EXP023/analyze.py --config experiments/EXP023/analysis.yaml
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

from src.fourier import frequency_energy  # noqa: E402
from src.dynamics import (alias_margin, early_topk_jaccard, freq_set, freq_trajectory, instability,  # noqa: E402
                          k_drops, read_metrics)
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def measure(rdir: Path, cfg: dict) -> dict:
    s = json.loads((rdir / "summary.json").read_text())
    mc = ModelConfig.from_dict(s["config"]["model"])
    p = s["config"]["task"]["p"]
    m = Transformer(mc)
    m.load_state_dict(torch.load(rdir / "checkpoints" / cfg["checkpoint"], map_location="cpu", weights_only=True))
    assert state_sha256(m) == s["final_state_sha256"], f"{rdir}: checkpoint hash mismatch"
    ks, sh = freq_set(m.W_E.detach(), p, cfg["major_frequency_min_share"])
    am = alias_margin(ks, p)
    E = frequency_energy(m.W_E.detach()[:p])
    spectrum = (E / E.sum()).tolist()                       # index 0 = const, k = frequency k
    traj = freq_trajectory(rdir / "checkpoints", p, cfg["major_frequency_min_share"])
    drops = k_drops(traj)
    rows = read_metrics(rdir / "metrics.csv")
    ins = instability(rows, cfg["instability"]["loss_threshold"], cfg["instability"]["buffer_steps"])
    thr, w = cfg["instability"]["loss_threshold"], cfg["explore"]["drop_spike_window"]
    spike_steps = [r["step"] for r in rows if not (r["train_loss"] < thr)]
    for d in drops:
        d["spike_within_window"] = any(abs(x - d["step"]) <= w for x in spike_steps)
    return {"n_layers": mc.n_layers, "data_seed": s["data_seed"], "test_acc": s["final"]["test_acc"],
            "first_test": s["first_step_test_acc_ge_0.99"], "freqs": ks, "freq_shares": sh,
            "K": len(ks), "spectrum": spectrum, "rel_margin": am["rel_margin"], "top_alias_offsets": [d for d, _ in am["top_offsets"]],
            "instability": ins, "K_trajectory": [(t["step"], t["K"]) for t in traj], "K_drops": drops,
            "early_topk_jaccard": early_topk_jaccard(traj, rdir / "checkpoints", p, cfg["explore"]["early_steps"])}


def rule(cases: list[bool], n_min: int) -> str:
    if len(cases) < n_min:
        return "UNDECIDABLE"
    return "MET" if all(cases) else "NOT MET"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    c26, c27 = cfg["H026"], cfg["H027"]

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    new = {f"seed{s}": measure(ROOT / cfg["runs_dir"] / f"seed{s}", cfg) for s in cfg["seeds"]}
    ref = {str(Path(r).parent.parent.name) + "/" + Path(r).name: measure(ROOT / r, cfg) for r in cfg["reference"]}
    for r in new.values():
        assert r["n_layers"] == 2 and r["data_seed"] == 598

    included = {k: r for k, r in new.items() if r["test_acc"] >= cfg["exclude_below_acc"]}
    excluded = sorted(set(new) - set(included))
    stuck = lambda r: r["test_acc"] < cfg["stuck_below_acc"]
    lo = [r for r in included.values() if r["rel_margin"] < c26["stuck_if_rel_margin_below"]]
    hi = [r for r in included.values() if r["rel_margin"] >= c26["ok_if_rel_margin_at_least"]]
    st = [r for r in included.values() if stuck(r)]
    k2 = [r for r in included.values() if r["K"] <= c27["max_K_stuck"]]
    verdicts = {
        "H026a (rel_margin < %.2f -> stuck)" % c26["stuck_if_rel_margin_below"]: rule([stuck(r) for r in lo], c26["min_models_a"]),
        "H026b (rel_margin >= %.2f -> not stuck)" % c26["ok_if_rel_margin_at_least"]: rule([not stuck(r) for r in hi], c26["min_models_b"]),
        "H027a (stuck -> K <= %d)" % c27["max_K_stuck"]: rule([r["K"] <= c27["max_K_stuck"] for r in st], c27["min_models"]),
        "H027b (K <= %d -> stuck)" % c27["max_K_stuck"]: rule([stuck(r) for r in k2], c27["min_models"]),
    }

    def show(name: str, r: dict) -> None:
        ins = r["instability"]
        print(f"  {name:14s} acc {r['test_acc']:.4f} {'STUCK' if stuck(r) else '     '} first {str(r['first_test']):>6s} "
              f"K {r['K']} {r['freqs']} m {r['rel_margin']:.3f} alias d {r['top_alias_offsets'][:2]} | "
              f"spikes {ins['events']} max {ins['max_train_loss']:.1e} | K drops "
              + (", ".join(f"{d['step']}:{d['K_from']}->{d['K_to']}{'*' if d['spike_within_window'] else ''}"
                           for d in r["K_drops"]) or "-")
              + " | early J " + " ".join(f"{k}:{'-' if v is None else f'{v:.2f}'}" for k, v in r["early_topk_jaccard"].items()))

    print("=== EXP023: residual error vs alias margin / number of frequencies (2-layer) ===")
    print("(* = K drop within the window of a training-loss spike)")
    print("FRESH models (verdict):")
    for k, r in new.items():
        show(k, r)
    print(f"excluded (test acc < {cfg['exclude_below_acc']}): {excluded or 'none'}")
    print("REFERENCE models (inspired the hypotheses; NOT in the verdict):")
    for k, r in ref.items():
        show(k, r)
    print(f"\ncounts: included {len(included)} | rel_margin<{c26['stuck_if_rel_margin_below']}: {len(lo)} | "
          f"rel_margin>={c26['ok_if_rel_margin_at_least']}: {len(hi)} | stuck: {len(st)} | K<={c27['max_K_stuck']}: {len(k2)}")
    print("\n=== PREDICTIONS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(), "verdicts": verdicts,
               "excluded": excluded, "runs": new, "reference": ref, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
