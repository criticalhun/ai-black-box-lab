"""
EXP001 training script (one seed per run).

    uv run python experiments/EXP001/train.py --config experiments/EXP001/config.yaml --seed 0
    uv run python experiments/EXP001/train.py --config experiments/EXP001/config.yaml --seed 0 --tag rerun
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from src.data import modular_addition  # noqa: E402
from src.model import ModelConfig, Transformer  # noqa: E402
from src.repro import setup_determinism, state_sha256  # noqa: E402

ACC_THRESHOLD = 0.99
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def git_info() -> dict:
    def run(*a: str) -> str:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def forward_loss(model, x, y, loss_dtype):
    logits = model(x)[:, -1, :].to(loss_dtype)
    return F.cross_entropy(logits, y), logits


def accuracy(logits, y) -> float:
    return (logits.argmax(dim=-1) == y).double().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    task, tr, lg = cfg["task"], cfg["training"], cfg["logging"]
    rt, rep = cfg["runtime"], cfg["reproducibility"]

    # --- determinism FIRST, before any CUDA work ---
    if rep["deterministic"]:
        setup_determinism(rep["cublas_workspace_config"], rep["float32_matmul_precision"])

    if args.seed not in cfg["seeds"]:
        sys.exit(f"seed {args.seed} not in preregistered seeds {cfg['seeds']}")
    git = git_info()
    if git["dirty"] and not args.allow_dirty:
        sys.exit("uncommitted changes in tracked files -> commit first (or --allow-dirty)")

    assert task["name"] == "modular_addition"
    assert tr["optimizer"] == "adamw" and tr["batch"] == "full" and tr["loss_position"] == -1
    assert rt["dtype"] == "float32"
    device = torch.device(rt["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("config requires CUDA, but CUDA is not available")
    loss_dtype = DTYPES[tr["loss_dtype"]]

    run_name = f"seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = args.config.parent / "results" / run_name
    if out.exists():
        sys.exit(f"{out} exists -> refusing to overwrite (use a new --tag)")
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    ds = modular_addition(task["p"], task["train_fraction"], task["data_seed"])
    x_tr, y_tr = (t.to(device) for t in ds.train)
    x_te, y_te = (t.to(device) for t in ds.test)

    torch.manual_seed(args.seed)
    model = Transformer(ModelConfig.from_dict(cfg["model"]))  # init on CPU
    init_hash = state_sha256(model)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"], betas=tuple(tr["betas"]))
    warm = tr["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 1.0))

    max_steps, log_every, ckpt_every = tr["max_steps"], lg["log_every"], lg["checkpoint_every"]
    first = {"train": None, "test": None}
    last = None

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print(f"{run_name}: params={model.n_params():,} train={len(y_tr)} test={len(y_te)} "
          f"split={ds.split_sha256()[:12]} device={device} ({gpu}) "
          f"commit={str(git['commit'])[:8]}", flush=True)
    t0 = time.time()

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "train_loss", "train_acc", "test_loss", "test_acc"])
        for step in range(max_steps + 1):
            train_loss, train_logits = forward_loss(model, x_tr, y_tr, loss_dtype)

            if step % log_every == 0 or step == max_steps:
                with torch.no_grad():
                    test_loss, test_logits = forward_loss(model, x_te, y_te, loss_dtype)
                last = {"step": step, "lr": sched.get_last_lr()[0],
                        "train_loss": train_loss.item(), "train_acc": accuracy(train_logits, y_tr),
                        "test_loss": test_loss.item(), "test_acc": accuracy(test_logits, y_te)}
                w.writerow(list(last.values()))
                f.flush()
                if first["train"] is None and last["train_acc"] >= ACC_THRESHOLD:
                    first["train"] = step
                if first["test"] is None and last["test_acc"] >= ACC_THRESHOLD:
                    first["test"] = step
                if step % (10 * log_every) == 0 or step == max_steps:
                    print(f"step {step:6d} | train loss {last['train_loss']:.3e} acc {last['train_acc']:.4f}"
                          f" | test loss {last['test_loss']:.3e} acc {last['test_acc']:.4f}"
                          f" | {time.time() - t0:5.0f}s", flush=True)

            if step % ckpt_every == 0 or step == max_steps:
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           ckpt_dir / f"step{step:06d}.pt")
            if step == max_steps:
                break

            opt.zero_grad(set_to_none=True)
            train_loss.backward()
            opt.step()
            sched.step()

    summary = {
        "experiment": cfg["experiment"], "hypothesis": cfg["hypothesis"], "run": run_name,
        "seed": args.seed, "data_seed": task["data_seed"], "split_sha256": ds.split_sha256(),
        "n_train": len(y_tr), "n_test": len(y_te), "n_params": model.n_params(),
        "max_steps": max_steps, "final": last,
        "first_step_train_acc_ge_0.99": first["train"],
        "first_step_test_acc_ge_0.99": first["test"],
        "init_state_sha256": init_hash,
        "final_state_sha256": state_sha256(model),
        "wall_seconds": round(time.time() - t0, 1),
        "git": git, "device": str(device), "gpu": gpu,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "config": cfg,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
