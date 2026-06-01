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
# Description:  Per-modality geometry encoders (kappa / c2 / H) for the PFV
#               presence classifier.
# -----------------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """log-magnitude with sign preserved; squashes wide-range integer values."""
    return torch.sign(x) * torch.log1p(x.abs())


class SetTransformerPool(nn.Module):
    """Permutation-invariant set pooling (Set Transformer, Lee et al. 2019).

    n_sab transformer encoder layers without positional encodings (so they are
    permutation-equivariant: every element attends to every other), then
    attention pooling -- n_seeds learned queries cross-attend over the set
    to give n_seeds fixed summary vectors, invariant to set order and size.
    Input (B, N, D) + padding mask -> output (B, n_seeds * D).
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_sab: int = 2,
                 n_seeds: int = 1, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.sab = nn.TransformerEncoder(layer, num_layers=n_sab, enable_nested_tensor=False)

        self.n_seeds = n_seeds
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        self.pma_norm_q = nn.LayerNorm(d_model)
        self.pma_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.pma_norm_o = nn.LayerNorm(d_model)
        self.pma_ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x   : (B, N, D)
        mask: (B, N) bool, True for valid tokens
        returns: (B, n_seeds * D)
        """
        kp = (~mask) if mask is not None else None
        # SAB: self-attention over the set
        x = self.sab(x, src_key_padding_mask=kp)

        # PMA: learned queries attend over the set
        B = x.shape[0]
        Q = self.seeds.unsqueeze(0).expand(B, -1, -1)             # (B, S, D)
        Qn = self.pma_norm_q(Q)
        pooled, _ = self.pma_attn(Qn, x, x, key_padding_mask=kp,
                                  need_weights=False)
        pooled = Q + pooled
        pooled = pooled + self.pma_ff(self.pma_norm_o(pooled))    # (B, S, D)
        return pooled.flatten(1)                                  # (B, S*D)


class KappaEncoder(nn.Module):
    """Set Transformer over symmetric-COO kappa tokens (i, j, k, value).

    Two nested symmetries. Within a token the three indices are unordered
    (kappa is fully symmetric), so they share one embedding table and are
    summed -- a Deep Sets aggregation over the triple that preserves index
    multiplicity. That sum is concatenated with the signed-log1p value (via a
    Linear), run through a per-token MLP, and the resulting unordered token set
    is pooled by the Set Transformer above.
    """

    def __init__(
        self,
        max_h11: int,
        d_idx: int = 32,
        d_val: int = 16,
        d_model: int = 128,
        d_out: int = 128,
        n_heads: int = 4,
        n_sab: int = 2,
        n_seeds: int = 1,
    ):
        super().__init__()
        self.max_h11 = max_h11
        self.idx_emb = nn.Embedding(max_h11, d_idx)
        self.val_proj = nn.Linear(1, d_val)

        d_in = d_idx + d_val
        # input projection (per-token MLP)
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # set transformer pooling
        self.pool = SetTransformerPool(
            d_model=d_model, n_heads=n_heads, n_sab=n_sab, n_seeds=n_seeds,
        )

        # output projection
        self.out = nn.Sequential(
            nn.Linear(n_seeds * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_out),
        )

    def forward(self, kappa_idx: torch.Tensor, kappa_v: torch.Tensor,
                kappa_mask: torch.Tensor) -> torch.Tensor:
        """
        kappa_idx: (B, N, 3) long
        kappa_v  : (B, N)    float
        kappa_mask: (B, N)   bool, True for real tokens
        returns: (B, d_out) float
        """
        # symmetric index aggregation: sum of three embeddings -> (B, N, d_idx)
        idx_emb = self.idx_emb(kappa_idx).sum(dim=-2)

        # value embedding: (B, N, 1) -> (B, N, d_val)
        v = signed_log1p(kappa_v).unsqueeze(-1)
        val_emb = self.val_proj(v)

        # per-token feature
        tok = torch.cat([idx_emb, val_emb], dim=-1)              # (B, N, d_in)
        tok = self.in_proj(tok)                                   # (B, N, d_model)

        # set transformer pooling (with padding mask)
        pooled = self.pool(tok, mask=kappa_mask)                  # (B, S*d_model)
        return self.out(pooled)                                   # (B, d_out)


class C2Encoder(nn.Module):
    """c2 is a length-h11 integer vector. Position matters (each entry is
    the second Chern class component for a specific basis divisor).

    Tokenization: each position j becomes a token
    E_pos[j] + Linear(signed_log1p(c2[j])). Aggregator: a small transformer
    encoder over the tokens, then mean-pool over valid positions.
    """
    def __init__(
        self,
        max_h11: int,
        d_pos: int = 32,
        d_val: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_out: int = 128,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_h11, d_pos)
        self.val_proj = nn.Linear(1, d_val)
        self.in_proj = nn.Linear(d_pos + d_val, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2*d_model,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, c2: torch.Tensor, c2_mask: torch.Tensor) -> torch.Tensor:
        """
        c2:      (B, H) long
        c2_mask: (B, H) bool, True at valid positions (h11 may differ)
        returns: (B, d_out)
        """
        B, H = c2.shape
        pos = torch.arange(H, device=c2.device).unsqueeze(0).expand(B, H)
        p = self.pos_emb(pos)                                       # (B, H, d_pos)
        v = signed_log1p(c2.float()).unsqueeze(-1)                  # (B, H, 1)
        v = self.val_proj(v)                                        # (B, H, d_val)
        tok = self.in_proj(torch.cat([p, v], dim=-1))               # (B, H, d_model)

        # transformer expects src_key_padding_mask: True = pad (i.e., ignore)
        pad = ~c2_mask
        z = self.enc(tok, src_key_padding_mask=pad)                 # (B, H, d_model)

        m = c2_mask.unsqueeze(-1).to(z.dtype)
        pooled = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.out(pooled)                                     # (B, d_out)


class HEncoder(nn.Module):
    """H is (n_rows, h11-1) integers. Rows are unordered (set of Mori cone
    hyperplanes, after dropping the conifold direction). Columns are ordered
    (each column is a specific direction in the cob basis).

    Per-row encoding: column-position embedding E_col[j] modulated by the
    signed-log1p value, summed over columns (handles variable h11-1).
    Set aggregator across rows: Deep Sets (sum-pool, 1/sqrt(count) rescaled).
    """
    def __init__(
        self,
        max_h11: int,
        d_col: int = 32,
        d_val: int = 16,
        d_model: int = 128,
        d_out: int = 128,
    ):
        super().__init__()
        # H has h11-1 columns; embed up to max_h11-1
        self.col_emb = nn.Embedding(max_h11 - 1, d_col)
        self.val_proj = nn.Linear(1, d_val)
        # per-(row, col) token feature dim
        self.row_phi = nn.Sequential(
            nn.Linear(d_col + d_val, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        # per-row (set element) feature MLP
        self.set_phi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        # post-pool MLP
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_out),
        )

    def forward(self, H: torch.Tensor, H_row_mask: torch.Tensor,
                H_col_mask: torch.Tensor) -> torch.Tensor:
        """
        H:          (B, M, C) long  - C = max_h11 - 1
        H_row_mask: (B, M)    bool  - which rows are valid
        H_col_mask: (B, C)    bool  - which columns are valid (= h11_b - 1)
        returns:    (B, d_out)
        """
        B, M, C = H.shape

        col_ids = torch.arange(C, device=H.device).view(1, 1, C).expand(B, M, C)
        col_e = self.col_emb(col_ids)                                # (B, M, C, d_col)
        v = signed_log1p(H.float()).unsqueeze(-1)                    # (B, M, C, 1)
        v = self.val_proj(v)                                         # (B, M, C, d_val)

        per_col = self.row_phi(torch.cat([col_e, v], dim=-1))        # (B, M, C, d_model)

        # sum over columns (only valid ones)
        col_mask = (H_row_mask.unsqueeze(-1) & H_col_mask.unsqueeze(1)).unsqueeze(-1)
        col_mask = col_mask.to(per_col.dtype)
        row_feat = (per_col * col_mask).sum(dim=2)                   # (B, M, d_model)
        n_cols = col_mask.sum(dim=2).clamp(min=1.0)                  # (B, M, 1)
        row_feat = row_feat / n_cols.sqrt()

        row_feat = self.set_phi(row_feat)                            # (B, M, d_model)

        # sum-pool valid rows
        rmask = H_row_mask.unsqueeze(-1).to(row_feat.dtype)
        pooled = (row_feat * rmask).sum(dim=1)                       # (B, d_model)
        n_rows = rmask.sum(dim=1).clamp(min=1.0)
        pooled = pooled / n_rows.sqrt()

        return self.rho(pooled)                                      # (B, d_out)
