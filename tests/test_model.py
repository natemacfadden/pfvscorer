"""Forward-pass smoke tests for the encoders and the presence classifier.

These build a small synthetic batch (no parquet needed) so they run anywhere.
"""

import torch

from pfvscorer import PFVRichnessModel, KappaEncoder, C2Encoder, HEncoder

B, MAX_H11 = 4, 10


def synthetic_batch(n_kappa=7, n_h_rows=5):
    h = MAX_H11
    return {
        "kappa_idx": torch.randint(0, h, (B, n_kappa, 3)),
        "kappa_v": torch.randn(B, n_kappa),
        "kappa_mask": torch.ones(B, n_kappa, dtype=torch.bool),
        "c2": torch.randint(-5, 5, (B, h)),
        "c2_mask": torch.ones(B, h, dtype=torch.bool),
        "H": torch.randint(-3, 3, (B, n_h_rows, h - 1)),
        "H_row_mask": torch.ones(B, n_h_rows, dtype=torch.bool),
        "H_col_mask": torch.ones(B, h - 1, dtype=torch.bool),
        "h11": torch.full((B,), h, dtype=torch.long),
        "h21": torch.randint(20, 200, (B,)).float(),
        "B": torch.randint(1, 50, (B,)).float(),
        "dil": torch.randint(1, 400, (B,)).float(),
    }


def test_encoders_output_shape():
    b = synthetic_batch()
    assert KappaEncoder(MAX_H11, d_out=128)(
        b["kappa_idx"], b["kappa_v"], b["kappa_mask"]
    ).shape == (B, 128)
    assert C2Encoder(MAX_H11, d_out=128)(b["c2"], b["c2_mask"]).shape == (B, 128)
    assert HEncoder(MAX_H11, d_out=128)(
        b["H"], b["H_row_mask"], b["H_col_mask"]
    ).shape == (B, 128)


def test_count_model_forward():
    model = PFVRichnessModel(max_h11=MAX_H11).eval()
    with torch.no_grad():
        logit = model(synthetic_batch())
        p = model.probs(synthetic_batch())
    assert logit.shape == (B,)
    assert p.shape == (B,)
    assert torch.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_multi_head():
    model = PFVRichnessModel(max_h11=MAX_H11, n_out=2).eval()
    with torch.no_grad():
        p = model.probs(synthetic_batch())
    assert p.shape == (B, 2) and (p >= 0).all() and (p <= 1).all()


def test_padding_invariance():
    """Adding masked-out padding tokens must not change the output."""
    model = PFVRichnessModel(max_h11=MAX_H11).eval()
    b = synthetic_batch(n_kappa=7, n_h_rows=5)
    with torch.no_grad():
        ref = model.probs(b)

    pad_k, pad_h = 3, 2
    bp = dict(b)
    bp["kappa_idx"] = torch.cat(
        [b["kappa_idx"], torch.randint(0, MAX_H11, (B, pad_k, 3))], dim=1
    )
    bp["kappa_v"] = torch.cat([b["kappa_v"], torch.randn(B, pad_k)], dim=1)
    bp["kappa_mask"] = torch.cat(
        [b["kappa_mask"], torch.zeros(B, pad_k, dtype=torch.bool)], dim=1
    )
    bp["H"] = torch.cat([b["H"], torch.randint(-3, 3, (B, pad_h, MAX_H11 - 1))], dim=1)
    bp["H_row_mask"] = torch.cat(
        [b["H_row_mask"], torch.zeros(B, pad_h, dtype=torch.bool)], dim=1
    )
    with torch.no_grad():
        out = model.probs(bp)
    assert torch.allclose(ref, out, atol=1e-5)
