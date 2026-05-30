"""Dataset and collate_fn for the encoded per-coni parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# repo_root/data/conis_encoded.parquet  (src/pfvscorer/dataset.py -> parents[2] = repo root)
DEFAULT_PARQUET = str(Path(__file__).resolve().parents[2] / "data" / "conis_encoded.parquet")


def random_unimodular_block(n: int, n_ops: int = 4, k_range: int = 2,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample V in GL(n, Z) by composing elementary integer matrices.

    n is the size of the block acting on directions 1..h11-1; the conifold
    direction (index 0) is left untouched so that coni_curve = (1, 0, ..., 0)
    is preserved.
    """
    if rng is None:
        rng = np.random.default_rng()
    V = np.eye(n, dtype=np.int64)
    if n < 2:
        return V
    for _ in range(n_ops):
        op = rng.integers(3)
        if op == 0:  # transvection: V[i, :] += k * V[j, :]
            i, j = rng.choice(n, 2, replace=False)
            k = int(rng.integers(-k_range, k_range + 1))
            if k != 0:
                E = np.eye(n, dtype=np.int64)
                E[i, j] = k
                V = E @ V
        elif op == 1:  # sign flip on one row
            i = int(rng.integers(n))
            E = np.eye(n, dtype=np.int64)
            E[i, i] = -1
            V = E @ V
        else:  # row permutation
            i, j = rng.choice(n, 2, replace=False)
            E = np.eye(n, dtype=np.int64)
            E[i, i] = 0
            E[j, j] = 0
            E[i, j] = 1
            E[j, i] = 1
            V = E @ V
    return V


def apply_basis_aug(kappa_coo, c2, H, h11: int, V: np.ndarray):
    """Apply U = [[1, 0], [0, V]] to (kappa, c2, H) at the dense level,
    then re-canonicalize kappa back to symmetric COO."""
    # build full U
    U = np.eye(h11, dtype=np.int64)
    U[1:, 1:] = V

    # dense kappa from canonical COO using symmetry
    dense = np.zeros((h11, h11, h11), dtype=np.int64)
    for i, j, k, v in kappa_coo:
        for perm in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            dense[perm] = v

    new_dense = np.einsum('ai,bj,ck,ijk->abc', U, U, U, dense)
    new_c2    = U @ np.asarray(c2, dtype=np.int64)
    new_H     = np.asarray(H, dtype=np.int64) @ V.T

    # back to canonical COO
    new_kappa = []
    for i in range(h11):
        for j in range(i, h11):
            for k in range(j, h11):
                v = int(new_dense[i, j, k])
                if v != 0:
                    new_kappa.append([i, j, k, v])
    return new_kappa, new_c2.tolist(), new_H.tolist()


class ConiDataset(Dataset):
    """One row per (h11, h21, polyID, classID, coniID).

    Each sample is a dict with tensors:
      kappa_idx : (N, 3) int64  - (i, j, k) indices, 0 <= i <= j <= k < h11
      kappa_v   : (N,)   float  - values (kappa[i,j,k])
      c2        : (h11,) int64
      H         : (M, h11-1) int64
      h11       : ()     int64
      num_pfvs  : ()     int64
    """

    def __init__(self, parquet_path: str = DEFAULT_PARQUET,
                 h11_filter: list[int] | None = None,
                 augment: bool = False,
                 aug_n_ops: int = 4,
                 aug_k_range: int = 2):
        df = pq.read_table(parquet_path).to_pandas()
        if h11_filter is not None:
            df = df[df.h11.isin(h11_filter)].reset_index(drop=True)
        self.df = df
        self.augment = augment
        self.aug_n_ops = aug_n_ops
        self.aug_k_range = aug_k_range

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        # pandas/pyarrow gives object arrays of lists for ragged columns; flatten to plain lists
        kappa_coo = [list(x) for x in r.kappa_coo]
        c2_list   = list(r.c2)
        H_list    = [list(x) for x in r.H]

        if self.augment:
            h11 = int(r.h11)
            V = random_unimodular_block(h11 - 1, n_ops=self.aug_n_ops, k_range=self.aug_k_range)
            kappa_coo, c2_list, H_list = apply_basis_aug(kappa_coo, c2_list, H_list, h11, V)

        if len(kappa_coo) == 0:
            kappa_idx = torch.zeros((0, 3), dtype=torch.long)
            kappa_v   = torch.zeros((0,),   dtype=torch.float32)
        else:
            arr = torch.tensor(kappa_coo, dtype=torch.long)
            kappa_idx = arr[:, :3].contiguous()
            kappa_v   = arr[:, 3].to(torch.float32)

        H_tensor = torch.tensor(H_list, dtype=torch.long) if len(H_list) > 0 else torch.zeros((0, 0), dtype=torch.long)

        return {
            'kappa_idx': kappa_idx,
            'kappa_v':   kappa_v,
            'c2':        torch.tensor(c2_list, dtype=torch.long),
            'H':         H_tensor,
            'h11':       torch.tensor(int(r.h11), dtype=torch.long),
            'num_pfvs':  torch.tensor(int(r.num_pfvs), dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    """Pad kappa tokens and H rows to max length in batch; pad c2 to max h11.

    Returns a dict of batched tensors plus boolean masks (True = valid).
    """
    B = len(batch)
    max_h11 = int(max(b['h11'].item() for b in batch))
    max_kappa = max(b['kappa_idx'].shape[0] for b in batch)
    max_h_rows = max(b['H'].shape[0] for b in batch)

    kappa_idx = torch.zeros((B, max_kappa, 3), dtype=torch.long)
    kappa_v   = torch.zeros((B, max_kappa),   dtype=torch.float32)
    kappa_mask = torch.zeros((B, max_kappa),  dtype=torch.bool)

    c2 = torch.zeros((B, max_h11), dtype=torch.long)
    c2_mask = torch.zeros((B, max_h11), dtype=torch.bool)

    h_cols = max_h11 - 1
    H = torch.zeros((B, max_h_rows, h_cols), dtype=torch.long)
    H_row_mask = torch.zeros((B, max_h_rows), dtype=torch.bool)
    H_col_mask = torch.zeros((B, h_cols), dtype=torch.bool)

    h11_t   = torch.zeros((B,), dtype=torch.long)
    pfvs_t  = torch.zeros((B,), dtype=torch.long)

    for b, s in enumerate(batch):
        n = s['kappa_idx'].shape[0]
        if n > 0:
            kappa_idx[b, :n] = s['kappa_idx']
            kappa_v[b, :n]   = s['kappa_v']
            kappa_mask[b, :n] = True

        h = int(s['h11'].item())
        c2[b, :h] = s['c2']
        c2_mask[b, :h] = True

        Hs = s['H']
        if Hs.numel() > 0:
            r_, c_ = Hs.shape
            H[b, :r_, :c_] = Hs
            H_row_mask[b, :r_] = True
            H_col_mask[b, :c_] = True

        h11_t[b]  = s['h11']
        pfvs_t[b] = s['num_pfvs']

    return {
        'kappa_idx':  kappa_idx,
        'kappa_v':    kappa_v,
        'kappa_mask': kappa_mask,
        'c2':         c2,
        'c2_mask':    c2_mask,
        'H':          H,
        'H_row_mask': H_row_mask,
        'H_col_mask': H_col_mask,
        'h11':        h11_t,
        'num_pfvs':   pfvs_t,
    }
