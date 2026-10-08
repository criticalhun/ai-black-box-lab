"""Sanity checks for src/data.py. Run: uv run python -m tests.test_data"""
import torch

from src.data import modular_addition


def main() -> None:
    p = 113
    ds = modular_addition(p, 0.3, 598)
    a, b, eq = ds.tokens.T

    assert ds.tokens.shape == (p * p, 3)
    assert torch.all(eq == p)
    assert torch.equal(ds.labels, (a + b) % p)
    assert ds.labels.max() < p                       # "=" is never a label
    i = 12 * p + 7
    assert ds.tokens[i].tolist() == [12, 7, p] and ds.labels[i].item() == 19
    assert ds.labels[100 * p + 20].item() == 7       # 120 mod 113

    n_tr, n_te = len(ds.train_idx), len(ds.test_idx)
    print(f"train/test: {n_tr}/{n_te}")
    assert n_tr == 3830 and n_tr + n_te == p * p
    # disjoint and complete
    assert torch.equal(torch.cat([ds.train_idx, ds.test_idx]).sort().values, torch.arange(p * p))
    # same data_seed -> same split; different -> different
    assert modular_addition(p, 0.3, 598).split_sha256() == ds.split_sha256()
    assert modular_addition(p, 0.3, 599).split_sha256() != ds.split_sha256()
    # split must NOT depend on the global torch seed (model-init seed)
    torch.manual_seed(12345)
    assert modular_addition(p, 0.3, 598).split_sha256() == ds.split_sha256()

    print(f"split sha256: {ds.split_sha256()[:16]}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
