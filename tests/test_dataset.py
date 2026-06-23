"""Tests for the dataset: signed-permutation augmentation and collate."""
import os

import numpy as np
import pytest
import torch

from pfvscorer import ConiDataset, collate
from pfvscorer.dataset import (
    DEFAULT_PARQUET,
    apply_basis_aug,
    random_signed_perm,
)


def test_random_signed_perm_is_signed_permutation():
    rng = np.random.default_rng(0)
    for n in range(1, 8):
        V = random_signed_perm(n, rng)
        assert V.shape == (n, n)
        assert round(abs(np.linalg.det(V))) == 1            # det = +-1
        # exactly one +-1 per row and per column (signed permutation)
        assert np.all(np.abs(V).sum(axis=0) == 1)
        assert np.all(np.abs(V).sum(axis=1) == 1)
        assert set(np.unique(V)).issubset({-1, 0, 1})


def test_identity_aug_is_noop():
    """A trivial basis change (V = I) must leave the canonical COO unchanged."""
    h11 = 4
    kappa_coo = [[0, 1, 2, 3], [1, 1, 3, -2], [0, 0, 0, 5]]
    c2 = [1, 2, 3, 4]
    H = [[1, 0, -1], [2, 1, 0]]
    V = np.eye(h11 - 1, dtype=np.int64)
    new_kappa, new_c2, new_H = apply_basis_aug(kappa_coo, c2, H, h11, V)
    assert sorted(new_kappa) == sorted(kappa_coo)
    assert new_c2 == c2
    assert new_H == H


@pytest.mark.skipif(not os.path.exists(DEFAULT_PARQUET),
                    reason="parquet not present")
def test_collate_shapes_and_masks():
    ds = ConiDataset(train=False)
    batch = collate([ds[i] for i in range(8)])

    B = 8
    assert batch['kappa_idx'].shape[0] == B
    assert batch['kappa_idx'].shape[-1] == 3
    # masks are boolean and agree with the padded dims
    assert batch['kappa_mask'].shape == batch['kappa_v'].shape
    assert batch['c2'].shape == batch['c2_mask'].shape
    assert batch['H'].shape[:2] == batch['H_row_mask'].shape
    # H has h11-1 columns
    assert batch['H'].shape[2] == batch['c2'].shape[1] - 1
    for key in ('h11', 'h21', 'B', 'dil', 'count'):
        assert batch[key].shape == (B,)


@pytest.mark.skipif(not os.path.exists(DEFAULT_PARQUET),
                    reason="parquet not present")
def test_conditioned_count_matches_filter():
    """The conditioned count equals the exact (B', dil') filter, and the
    label-preserving signed-perm aug leaves (B, dil, count) unchanged."""
    ds = ConiDataset(train=False)                 # deterministic sampling
    ds_aug = ConiDataset(train=False, augment=True)
    for i in (0, 1, 2, 100, 1000):
        s = ds[i]
        r = ds.df.iloc[i]
        infn = np.asarray(r.pfv_infnorm); rdil = np.asarray(r.pfv_reqdil)
        B, D = int(s['B']), int(s['dil'])
        manual = int(((infn <= B) & (rdil <= D)).sum()) if infn.size else 0
        assert int(s['count']) == manual
        sa = ds_aug[i]                            # same rng draw -> same window/count
        assert (int(sa['B']), int(sa['dil']), int(sa['count'])) == (B, D, manual)
