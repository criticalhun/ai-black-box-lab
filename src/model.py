"""
src/model.py — Minimal, fully hookable Transformer for interpretability.

    x0      = W_E[tokens] + W_pos
    x_mid   = x0 + Attn(x0)
    x_post  = x_mid + MLP(x_mid)
    logits  = x_post @ W_U

Only the configuration used in our experiments is implemented; any other
option fails loudly instead of silently doing something else.
Weights are initialized on CPU (move to GPU afterwards) so the same seed
gives the same initial weights on every device.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

HookFn = Callable[[torch.Tensor, str], "torch.Tensor | None"]


@dataclass
class ModelConfig:
    d_vocab: int = 114
    n_ctx: int = 3
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    n_layers: int = 1
    activation: str = "relu"
    layernorm: bool = False
    attn_bias: bool = False
    mlp_bias: bool = True
    causal_mask: bool = True
    init: str = "normal_std_1_over_sqrt_d_model"

    def __post_init__(self) -> None:
        bad = []
        if self.layernorm:
            bad.append("layernorm=True")
        if self.activation != "relu":
            bad.append(f"activation={self.activation}")
        if self.attn_bias:
            bad.append("attn_bias=True")
        if not self.mlp_bias:
            bad.append("mlp_bias=False")
        if self.init != "normal_std_1_over_sqrt_d_model":
            bad.append(f"init={self.init}")
        if bad:
            raise NotImplementedError("not implemented: " + ", ".join(bad))

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)  # unknown keys -> TypeError (catches config typos)

    def to_dict(self) -> dict:
        return asdict(self)


class HookPoint(nn.Module):
    """Identity layer. Exists only so activations can be read/modified by name."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        H, D, Dh = cfg.n_heads, cfg.d_model, cfg.d_head
        self.W_Q = nn.Parameter(torch.empty(H, D, Dh))
        self.W_K = nn.Parameter(torch.empty(H, D, Dh))
        self.W_V = nn.Parameter(torch.empty(H, D, Dh))
        self.W_O = nn.Parameter(torch.empty(H, Dh, D))
        ones = torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool)
        mask = torch.tril(ones) if cfg.causal_mask else ones
        self.register_buffer("mask", mask, persistent=False)

        self.hook_q = HookPoint()            # [batch, pos, head, d_head]
        self.hook_k = HookPoint()            # [batch, pos, head, d_head]
        self.hook_v = HookPoint()            # [batch, pos, head, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head, q_pos, k_pos] masked, pre-softmax
        self.hook_pattern = HookPoint()      # [batch, head, q_pos, k_pos]
        self.hook_z = HookPoint()            # [batch, pos, head, d_head]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        q = self.hook_q(torch.einsum("bpd,hde->bphe", x, self.W_Q))
        k = self.hook_k(torch.einsum("bpd,hde->bphe", x, self.W_K))
        v = self.hook_v(torch.einsum("bpd,hde->bphe", x, self.W_V))

        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(self.cfg.d_head)
        scores = scores.masked_fill(~self.mask[:n, :n], float("-inf"))
        scores = self.hook_attn_scores(scores)
        pattern = self.hook_pattern(scores.softmax(dim=-1))

        z = self.hook_z(torch.einsum("bhqk,bkhe->bqhe", pattern, v))
        return torch.einsum("bqhe,hed->bqd", z, self.W_O)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.b_in = nn.Parameter(torch.empty(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        self.b_out = nn.Parameter(torch.empty(cfg.d_model))
        self.hook_pre = HookPoint()   # [batch, pos, d_mlp] before ReLU
        self.hook_post = HookPoint()  # [batch, pos, d_mlp] after ReLU ("neurons")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.hook_pre(x @ self.W_in + self.b_in)
        post = self.hook_post(F.relu(pre))
        return post @ self.W_out + self.b_out


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.hook_resid_pre = HookPoint()   # [batch, pos, d_model]
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        attn_out = self.hook_attn_out(self.attn(x))
        x = self.hook_resid_mid(x + attn_out)
        mlp_out = self.hook_mlp_out(self.mlp(x))
        return self.hook_resid_post(x + mlp_out)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.W_U = nn.Parameter(torch.empty(cfg.d_model, cfg.d_vocab))
        self.hook_embed = HookPoint()       # [batch, pos, d_model]
        self.hook_pos_embed = HookPoint()   # [batch, pos, d_model]
        self._init_weights()

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.cfg.d_model)
        for name, p in self.named_parameters():
            if name.split(".")[-1].startswith("b_"):
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [batch, pos] (long) -> logits: [batch, pos, d_vocab]"""
        B, n = tokens.shape
        x = self.hook_embed(self.W_E[tokens])
        x = x + self.hook_pos_embed(self.W_pos[:n].unsqueeze(0).expand(B, -1, -1))
        for block in self.blocks:
            x = block(x)
        return x @ self.W_U

    # ---------- interpretability utilities ----------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def hook_points(self) -> dict[str, HookPoint]:
        return {n: m for n, m in self.named_modules() if isinstance(m, HookPoint)}

    @contextmanager
    def hooks(self, fwd_hooks: list[tuple[str, HookFn]]):
        """Temporarily attach hooks. fn(act, name) returns a replacement tensor or None."""
        points = self.hook_points()
        handles = []
        try:
            for name, fn in fwd_hooks:
                if name not in points:
                    raise KeyError(f"unknown hook point: {name}")

                def _wrap(_mod, _inp, out, fn=fn, name=name):
                    return fn(out, name)

                handles.append(points[name].register_forward_hook(_wrap))
            yield self
        finally:
            for h in handles:
                h.remove()

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks: list[tuple[str, HookFn]]) -> torch.Tensor:
        with self.hooks(fwd_hooks):
            return self(tokens)

    def run_with_cache(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cache: dict[str, torch.Tensor] = {}

        def save(act: torch.Tensor, name: str) -> None:
            cache[name] = act.detach().clone()

        with self.hooks([(n, save) for n in self.hook_points()]):
            logits = self(tokens)
        return logits, cache
