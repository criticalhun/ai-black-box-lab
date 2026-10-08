"""src/data.py — Controlled datasets. Index of pair (a, b) is a*p + b."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass
class SplitDataset:
    tokens: torch.Tensor     # [N, 3] long: [a, b, "="]
    labels: torch.Tensor     # [N] long
    train_idx: torch.Tensor  # sorted
    test_idx: torch.Tensor   # sorted

    @property
    def train(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[self.train_idx], self.labels[self.train_idx]

    @property
    def test(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[self.test_idx], self.labels[self.test_idx]

    def split_sha256(self) -> str:
        return hashlib.sha256(self.train_idx.numpy().tobytes()).hexdigest()


def modular_addition(p: int, train_fraction: float, data_seed: int) -> SplitDataset:
    a = torch.arange(p).repeat_interleave(p)   # 0,0,...,0,1,1,...
    b = torch.arange(p).repeat(p)              # 0,1,...,p-1,0,1,...
    eq = torch.full_like(a, p)                 # token p = "="
    tokens = torch.stack([a, b, eq], dim=1)
    labels = (a + b) % p

    # own CPU generator: split is independent of the global (model-init) seed
    g = torch.Generator().manual_seed(data_seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(train_fraction * p * p)
    return SplitDataset(tokens, labels,
                        perm[:n_train].sort().values,
                        perm[n_train:].sort().values)
