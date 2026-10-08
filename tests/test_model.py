"""Sanity checks for src/model.py. Run: uv run python -m tests.test_model"""
import torch
import torch.nn.functional as F

from src.repro import setup_determinism, state_sha256

setup_determinism()  # before any CUDA work

from src.model import ModelConfig, Transformer  # noqa: E402

R = "blocks.0."
CFG = ModelConfig()


def train_steps_hash(device: str, n_steps: int = 5) -> str:
    """Init on CPU (seed 0), move to device, run real AdamW steps, return weight hash."""
    torch.manual_seed(0)
    model = Transformer(CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 113, (256, 3), generator=g)
    x[:, 2] = 113
    y = (x[:, 0] + x[:, 1]) % 113
    x, y = x.to(device), y.to(device)
    for _ in range(n_steps):
        loss = F.cross_entropy(model(x)[:, -1, :].to(torch.float64), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return state_sha256(model)


def main() -> None:
    torch.manual_seed(0)
    model = Transformer(CFG)

    # 0. parameter count matches hand calculation
    n = model.n_params()
    print(f"params: {n:,}")
    assert n == 226_816, n

    tokens = torch.tensor([[12, 7, 113], [100, 20, 113]])  # 113 = "="
    logits, cache = model.run_with_cache(tokens)
    assert logits.shape == (2, 3, 114)
    print(f"hook points: {len(cache)}")
    for k, v in cache.items():
        print(f"  {k:30s} {tuple(v.shape)}")

    # 1. attention rows are probability distributions
    pat = cache[R + "attn.hook_pattern"]
    assert torch.allclose(pat.sum(-1), torch.ones_like(pat.sum(-1)))
    # 2. causal mask: no attention to future positions
    assert torch.all(pat.triu(diagonal=1) == 0)
    # 3. residual stream is an exact sum of component outputs
    recon = (cache["hook_embed"] + cache["hook_pos_embed"]
             + cache[R + "hook_attn_out"] + cache[R + "hook_mlp_out"])
    assert torch.allclose(recon, cache[R + "hook_resid_post"], atol=1e-6)
    # 4. logits are a linear readout of the final residual
    assert torch.allclose(logits, cache[R + "hook_resid_post"] @ model.W_U, atol=1e-5)
    # 5. an intervention changes the output ...
    zero = lambda act, name: torch.zeros_like(act)
    ablated = model.run_with_hooks(tokens, [(R + "hook_mlp_out", zero)])
    assert not torch.allclose(ablated, logits)
    # ... exactly as predicted (MLP term removed from the residual) ...
    assert torch.allclose(ablated, cache[R + "hook_resid_mid"] @ model.W_U, atol=1e-5)
    # ... and hooks do not leak afterwards
    assert torch.equal(model(tokens), logits), "hook leaked"
    # 6. same seed -> identical initial weights
    torch.manual_seed(0)
    assert state_sha256(Transformer(CFG)) == state_sha256(model)
    # 7. unsupported options fail loudly
    try:
        ModelConfig(layernorm=True)
        raise AssertionError("layernorm=True should fail")
    except NotImplementedError:
        pass
    print("CPU checks: OK")

    # 8. training determinism (forward + backward + AdamW), CPU
    h_cpu = train_steps_hash("cpu")
    assert h_cpu == train_steps_hash("cpu"), "CPU training not deterministic"
    print(f"CPU 5-step train hash: {h_cpu[:16]} (reproducible)")

    if torch.cuda.is_available():
        # 9. same model on GPU gives (numerically) the same outputs
        m_gpu = model.to("cuda")
        diff = (m_gpu(tokens.cuda()).cpu() - logits).abs().max().item()
        print(f"max |logits_gpu - logits_cpu|: {diff:.2e}")
        assert diff < 1e-4, diff
        # 10. training determinism on GPU
        h_gpu = train_steps_hash("cuda")
        assert h_gpu == train_steps_hash("cuda"), "GPU training not deterministic"
        print(f"GPU 5-step train hash: {h_gpu[:16]} (reproducible)")
        print(f"CPU vs GPU hash identical: {h_cpu == h_gpu} (expected False; not required)")
    else:
        print("CUDA not available -> GPU checks SKIPPED")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
