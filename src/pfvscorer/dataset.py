# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Dataset + collate for coni_pfvs.parquet. __getitem__ samples a
#               window (B', dil') and returns its exact PFV count. Schema:
#               huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs
# -----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# repo_root/data/coni_pfvs.parquet  (this file -> parents[2] = repo root)
DEFAULT_PARQUET = str(
    Path(__file__).resolve().parents[2] / "data" / "coni_pfvs.parquet"
)


def random_signed_perm(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a signed permutation matrix in GL(n, Z) (orthogonal, det +/-1).

    Acts on directions 1..h11-1; the conifold direction (index 0) is left
    untouched so coni_curve = (1, 0, ..., 0) is preserved. Signed permutations
    are the subgroup that preserves the L-infinity search box, keeping the
    per-(B, dilation) PFV counts exactly invariant.

    Parameters
    ----------
    n : int
        Block size (h11 - 1).
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Integer signed-permutation matrix, shape (n, n).
    """
    V = np.zeros((n, n), dtype=np.int64)
    if n < 1:
        return V
    perm = rng.permutation(n)
    signs = rng.choice(np.array([-1, 1], dtype=np.int64), size=n)
    V[np.arange(n), perm] = signs
    return V


def apply_basis_aug(kappa_coo: list, c2: list, H: list, h11: int, V: np.ndarray):
    """Apply the basis change U = [[1, 0], [0, V]] to (kappa, c2, H).

    kappa is expanded to a dense symmetric tensor, transformed by U, and
    re-canonicalized to upper-triangular COO.

    Parameters
    ----------
    kappa_coo : list of [i, j, k, value]
        Sparse symmetric triple-intersection numbers.
    c2 : list of int
        Second Chern class (length h11).
    H : list of list of int
        Hyperplane rows (each length h11 - 1).
    h11 : int
        Hodge number h^{1,1}.
    V : np.ndarray
        Signed-permutation block acting on directions 1..h11-1.

    Returns
    -------
    tuple
        (new_kappa_coo, new_c2, new_H), same formats as the inputs.
    """
    U = np.eye(h11, dtype=np.int64)
    U[1:, 1:] = V

    dense = np.zeros((h11, h11, h11), dtype=np.int64)
    for i, j, k, v in kappa_coo:
        for perm in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            dense[perm] = v

    new_dense = np.einsum("ai,bj,ck,ijk->abc", U, U, U, dense)
    new_c2 = U @ np.asarray(c2, dtype=np.int64)
    new_H = np.asarray(H, dtype=np.int64) @ V.T

    new_kappa = []
    for i in range(h11):
        for j in range(i, h11):
            for k in range(j, h11):
                v = int(new_dense[i, j, k])
                if v != 0:
                    new_kappa.append([i, j, k, v])
    return new_kappa, new_c2.tolist(), new_H.tolist()


class ConiDataset(Dataset):
    """One row per conifold; __getitem__ draws one (B', dil', count) sample.

    Returns a dict of tensors:
      kappa_idx : (N, 3) int64
      kappa_v   : (N,)   float
      c2        : (h11,) int64
      H         : (M, h11-1) int64
      h11       : ()     int64
      B         : ()     float  - sampled box bound B'
      dil       : ()     float  - sampled dilation dil'
      count     : ()     int64  - exact #PFVs within (B', dil')

    train=True: fresh randomness each epoch (augmentation). train=False:
    deterministic per-index sampling (reproducible eval) and no basis aug.
    """

    def __init__(
        self,
        parquet_path: str = DEFAULT_PARQUET,
        h11_filter: list[int] | None = None,
        augment: bool = False,
        train: bool = True,
        eval_seed: int = 12345,
    ):
        df = pq.read_table(parquet_path).to_pandas()
        if h11_filter is not None:
            df = df[df.h11.isin(h11_filter)].reset_index(drop=True)
        self.df = df
        self.augment = augment
        self.train = train
        self.eval_seed = eval_seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        if self.train:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.eval_seed + idx)

        # sample a window (B', dil') jointly from one area-weighted frontier
        fr_B = np.maximum(np.asarray(r.frontier_infnorm, dtype=np.int64), 1)
        fr_D = np.maximum(np.asarray(r.frontier_dil, dtype=np.int64), 1)
        areas = (fr_B * fr_D).astype(np.float64)
        fi = int(rng.choice(len(fr_B), p=areas / areas.sum()))
        Bp = int(rng.integers(1, int(fr_B[fi]) + 1))
        Dp = int(rng.integers(1, int(fr_D[fi]) + 1))

        infn = np.asarray(r.pfv_infnorm, dtype=np.int64)
        rdil = np.asarray(r.pfv_reqdil, dtype=np.int64)
        count = int(np.count_nonzero((infn <= Bp) & (rdil <= Dp))) if infn.size else 0

        # features (+ optional signed-perm basis augmentation)
        kappa_coo = [list(x) for x in r.kappa_coo]
        c2_list = list(r.c2)
        H_list = [list(x) for x in r.H]
        if self.augment:
            h11 = int(r.h11)
            V = random_signed_perm(h11 - 1, rng)
            kappa_coo, c2_list, H_list = apply_basis_aug(
                kappa_coo, c2_list, H_list, h11, V
            )

        if len(kappa_coo) == 0:
            kappa_idx = torch.zeros((0, 3), dtype=torch.long)
            kappa_v = torch.zeros((0,), dtype=torch.float32)
        else:
            arr = torch.tensor(kappa_coo, dtype=torch.long)
            kappa_idx = arr[:, :3].contiguous()
            kappa_v = arr[:, 3].to(torch.float32)

        if len(H_list) > 0:
            H_tensor = torch.tensor(H_list, dtype=torch.long)
        else:
            H_tensor = torch.zeros((0, 0), dtype=torch.long)

        return {
            "kappa_idx": kappa_idx,
            "kappa_v": kappa_v,
            "c2": torch.tensor(c2_list, dtype=torch.long),
            "H": H_tensor,
            "h11": torch.tensor(int(r.h11), dtype=torch.long),
            "h21": torch.tensor(float(r.h21), dtype=torch.float32),
            "B": torch.tensor(float(Bp), dtype=torch.float32),
            "dil": torch.tensor(float(Dp), dtype=torch.float32),
            "count": torch.tensor(count, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    """Collate per-coni samples into padded, masked batch tensors.

    Parameters
    ----------
    batch : list of dict
        Samples from ConiDataset.__getitem__.

    Returns
    -------
    dict
        Batched tensors (kappa/c2/H padded to the batch max) plus boolean masks
        (True = valid), and h11/h21/B/dil/count.
    """
    bs = len(batch)
    max_h11 = int(max(b["h11"].item() for b in batch))
    max_kappa = max(b["kappa_idx"].shape[0] for b in batch)
    max_h_rows = max(b["H"].shape[0] for b in batch)

    kappa_idx = torch.zeros((bs, max_kappa, 3), dtype=torch.long)
    kappa_v = torch.zeros((bs, max_kappa), dtype=torch.float32)
    kappa_mask = torch.zeros((bs, max_kappa), dtype=torch.bool)

    c2 = torch.zeros((bs, max_h11), dtype=torch.long)
    c2_mask = torch.zeros((bs, max_h11), dtype=torch.bool)

    h_cols = max_h11 - 1
    H = torch.zeros((bs, max_h_rows, h_cols), dtype=torch.long)
    H_row_mask = torch.zeros((bs, max_h_rows), dtype=torch.bool)
    H_col_mask = torch.zeros((bs, h_cols), dtype=torch.bool)

    h11_t = torch.zeros((bs,), dtype=torch.long)
    h21_t = torch.zeros((bs,), dtype=torch.float32)
    B_t = torch.zeros((bs,), dtype=torch.float32)
    dil_t = torch.zeros((bs,), dtype=torch.float32)
    cnt_t = torch.zeros((bs,), dtype=torch.long)

    for b, s in enumerate(batch):
        n = s["kappa_idx"].shape[0]
        if n > 0:
            kappa_idx[b, :n] = s["kappa_idx"]
            kappa_v[b, :n] = s["kappa_v"]
            kappa_mask[b, :n] = True

        h = int(s["h11"].item())
        c2[b, :h] = s["c2"]
        c2_mask[b, :h] = True

        Hs = s["H"]
        if Hs.numel() > 0:
            r_, c_ = Hs.shape
            H[b, :r_, :c_] = Hs
            H_row_mask[b, :r_] = True
            H_col_mask[b, :c_] = True

        h11_t[b] = s["h11"]
        h21_t[b] = s["h21"]
        B_t[b] = s["B"]
        dil_t[b] = s["dil"]
        cnt_t[b] = s["count"]

    return {
        "kappa_idx": kappa_idx,
        "kappa_v": kappa_v,
        "kappa_mask": kappa_mask,
        "c2": c2,
        "c2_mask": c2_mask,
        "H": H,
        "H_row_mask": H_row_mask,
        "H_col_mask": H_col_mask,
        "h11": h11_t,
        "h21": h21_t,
        "B": B_t,
        "dil": dil_t,
        "count": cnt_t,
    }
