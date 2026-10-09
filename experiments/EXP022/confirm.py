"""
EXP022: H025 — are the wrong answers of the 2-frequency model (EXP017 seed3, freqs {28, 43})
concentrated at the alias offset d = ±8, where both clocks almost return to their start?
Prediction, thresholds and this code were committed before the errors were ever inspected.

    uv run python experiments/EXP022/confirm.py --config experiments/EXP022/config.yaml
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
from src.dynamics import alias_margin, error_offsets, freq_set, share_in  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def judge(share: float, thr: float, n_wrong: int, min_wrong: int, valid: bool) -> str:
    if not valid:
        return "INVALID"
    if n_wrong < min_wrong:
        return "UNDECIDABLE"
    return "MET" if share >= thr else "NOT MET"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    src, c25 = cfg["source"], cfg["H025"]
    assert cfg["evaluation"] == {"inputs": "all_pairs", "error_split": "test", "device": "cpu"}

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

    res = {}
    for run in src["runs"]:
        rdir = exp / "results" / run
        s = json.loads((rdir / "summary.json").read_text())
        m = Transformer(model_cfg)
        m.load_state_dict(torch.load(rdir / "checkpoints" / src["checkpoint"], map_location="cpu", weights_only=True))
        assert state_sha256(m) == s["final_state_sha256"], f"{run}: checkpoint hash mismatch"
        ks, sh = freq_set(m.W_E.detach(), p, cfg["major_frequency_min_share"])
        am = alias_margin(ks, p)
        with torch.no_grad():
            lg = m(tokens)[:, -1, :].to(torch.float64)
        pred = lg.argmax(-1)
        e = error_offsets(pred[test_idx], tokens[test_idx], labels[test_idx], p)
        # exploratory: where is the runner-up answer, for ALL test pairs (also the correct ones)?
        lt = lg[test_idx].clone()
        y = labels[test_idx]
        correct_logit = lt.gather(1, y[:, None]).squeeze(1)
        lt.scatter_(1, y[:, None], float("-inf"))
        runner = lt.argmax(-1)
        gap = correct_logit - lt.max(-1).values
        rdelta = (runner - (tokens[test_idx, 0] + tokens[test_idx, 1])) % p
        rcounts = torch.bincount(rdelta, minlength=p)
        top_alias = {am["top_offsets"][0][0], am["top_offsets"][1][0]} if ks else set()
        res[run] = {
            "freqs": ks, "freq_shares": sh, "alias": am,
            "test_acc": 1 - e["n_wrong"] / e["n"], "errors": e,
            "explore": {"runner_up_share_at_own_top_alias": (rcounts[sorted(top_alias)].sum() / len(y)).item()
                        if top_alias else None,
                        "runner_up_top_offsets": [(int(k), int(rcounts[k])) for k in torch.argsort(rcounts, descending=True)[:4]],
                        "gap_min": gap.min().item(), "gap_median": gap.median().item()},
        }

    f = res[src["focus_run"]]
    valid = f["freqs"] == cfg["expected_focus_freqs"] and f["test_acc"] >= cfg["baseline_min_acc"]
    s1 = share_in(f["errors"]["counts"], set(c25["alias_offsets"]))
    s2 = share_in(f["errors"]["counts"], set(c25["secondary_offsets"]))
    verdicts = {
        "H025 primary": judge(s1, c25["min_share"], f["errors"]["n_wrong"], c25["min_wrong"], valid),
        "H025 secondary": judge(s2, c25["secondary_min_share"], f["errors"]["n_wrong"], c25["min_wrong"], valid),
    }

    print("=== EXP022: H025 — error offsets of the 2-frequency model ===")
    for run, r in res.items():
        a = r["alias"]
        print(f"\n##### {run}  freqs {r['freqs']}  test acc {r['test_acc']:.4f}  wrong {r['errors']['n_wrong']}"
              f"  alias rel_margin {a['rel_margin']:.3f}  top alias d: {[d for d, _ in a['top_offsets']]}")
        top = sorted(r["errors"]["counts"].items(), key=lambda kv: -kv[1])[:6]
        print(f"  wrong-answer offsets d=(pred-(a+b)) mod p, top: {top}")
        x = r["explore"]
        print(f"  [explore] runner-up offsets (all test pairs): {x['runner_up_top_offsets']} | "
              f"share at own top alias {x['runner_up_share_at_own_top_alias']}")
        print(f"  [explore] logit gap correct - best other: min {x['gap_min']:+.3f}, median {x['gap_median']:+.3f}")
    print(f"\nfocus {src['focus_run']}: freqs {f['freqs']} (expected {cfg['expected_focus_freqs']}), "
          f"test acc {f['test_acc']:.4f} (min {cfg['baseline_min_acc']}) -> valid={valid}")
    print(f"share of wrong answers at d in {c25['alias_offsets']}: {s1:.3f} (threshold {c25['min_share']})")
    print(f"share at d in {c25['secondary_offsets']}: {s2:.3f} (threshold {c25['secondary_min_share']})")
    print("\n=== PREDICTIONS ===")
    for h, v in verdicts.items():
        print(f"{h}: {v}")

    summary = {"experiment": cfg["experiment"], "git_commit": git_commit(), "verdicts": verdicts,
               "focus_shares": {"primary": s1, "secondary": s2}, "runs": res, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
