"""
EXP002: Fourier concentration of W_E in EXP001 models (tests H002).

    uv run python experiments/EXP002/analyze.py --config experiments/EXP002/config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.fourier import frequency_energy, top_k_share  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import state_sha256  # noqa: E402


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_model(path: Path, model_cfg: ModelConfig) -> Transformer:
    model = Transformer(model_cfg)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def analyze_we(model: Transformer, rows: list[int], k: int) -> tuple[dict, torch.Tensor]:
    E = frequency_energy(model.W_E[rows[0]:rows[1]])
    share, freqs = top_k_share(E, k)
    return {"top_k_share": share, "top_k_freqs": freqs,
            "const_share": (E[0] / E.sum()).item()}, E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    src, an, ex = cfg["source"], cfg["analysis"], cfg["exploratory"]
    assert an["matrix"] == "W_E" and an["basis"] == "real_fourier_orthonormal"
    k, rows = an["top_k"], an["rows"]
    thr, thr_ctrl = an["threshold_final"], an["threshold_control_max"]

    out = args.config.parent / "results"
    if (out / "summary.json").exists():
        sys.exit(f"{out}/summary.json exists -> refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    exp1 = ROOT / "experiments" / src["experiment"]
    exp1_cfg = yaml.safe_load((exp1 / "config.yaml").read_text())
    model_cfg = ModelConfig.from_dict(exp1_cfg["model"])
    runs = src["runs"]

    # ---------------- preregistered test ----------------
    per_run = {}
    for run in runs:
        rdir = exp1 / "results" / run
        s1 = json.loads((rdir / "summary.json").read_text())
        final_m = load_model(rdir / "checkpoints" / src["final_checkpoint"], model_cfg)
        ctrl_m = load_model(rdir / "checkpoints" / src["control_checkpoint"], model_cfg)
        # integrity: exactly the models EXP001 produced
        assert state_sha256(final_m) == s1["final_state_sha256"], f"{run}: final checkpoint hash mismatch"
        assert state_sha256(ctrl_m) == s1["init_state_sha256"], f"{run}: init checkpoint hash mismatch"

        final, E = analyze_we(final_m, rows, k)
        control, _ = analyze_we(ctrl_m, rows, k)
        frac = E / E.sum()
        top10 = torch.topk(frac[1:], 10)
        per_run[run] = {
            "final": final, "control": control,
            "final_top10_freq_share": [[int(i) + 1, round(v.item(), 4)]
                                       for v, i in zip(top10.values, top10.indices)],
        }

    finals = [per_run[r]["final"]["top_k_share"] for r in runs]
    controls = [per_run[r]["control"]["top_k_share"] for r in runs]
    n_pass = sum(f >= thr for f in finals)
    if not all(c < thr_ctrl for c in controls):
        verdict = "INVALID"
    elif n_pass == len(runs):
        verdict = "SUPPORTED"
    elif n_pass >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "REJECTED"

    print("=== H002: preregistered test ===")
    print(f"criterion: final top-{k} share >= {thr} (all runs), control < {thr_ctrl}")
    for r in runs:
        f, c = per_run[r]["final"], per_run[r]["control"]
        print(f"{r:6s} control {c['top_k_share']:.3f} | final {f['top_k_share']:.3f} "
              f"| const {f['const_share']:.3f} | top-{k} freqs {f['top_k_freqs']}")
    print(f"VERDICT: {verdict} ({n_pass}/{len(runs)} runs >= {thr})")

    # ---------------- exploratory (NOT part of the verdict) ----------------
    metrics = {}
    for run in runs:
        with open(exp1 / "results" / run / "metrics.csv") as fh:
            metrics[run] = {int(row["step"]): row for row in csv.DictReader(fh)}
    traj = []
    for step in range(0, exp1_cfg["training"]["max_steps"] + 1, ex["trajectory_every"]):
        row = {"step": step}
        for run in runs:
            m = load_model(exp1 / "results" / run / "checkpoints" / f"step{step:06d}.pt", model_cfg)
            row[f"{run}_share"] = analyze_we(m, rows, k)[0]["top_k_share"]
            row[f"{run}_test_acc"] = float(metrics[run][step]["test_acc"])
            row[f"{run}_test_loss"] = float(metrics[run][step]["test_loss"])
        traj.append(row)
    with open(out / "trajectory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(traj[0].keys()))
        w.writeheader()
        w.writerows(traj)

    print("\n=== EXPLORATORY: top-k share during training (not part of verdict) ===")
    print("  step | " + " | ".join(f"{r}: share  test_acc" for r in runs))
    for row in traj:
        if row["step"] % 2000 == 0:
            print(f"{row['step']:6d} | " + " | ".join(
                f"{r}: {row[f'{r}_share']:.3f}  {row[f'{r}_test_acc']:.3f}  " for r in runs))

    summary = {"experiment": cfg["experiment"], "hypothesis": cfg["hypothesis"],
               "git_commit": git_commit(), "verdict": verdict, "n_pass": n_pass,
               "threshold_final": thr, "threshold_control_max": thr_ctrl, "top_k": k,
               "runs": per_run, "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
