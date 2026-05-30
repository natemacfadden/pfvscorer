"""PFV count model: 3 encoders + h11 embedding -> MLP trunk -> log_lambda."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import KappaEncoder, C2Encoder, HEncoder


class PFVCountModel(nn.Module):
    """Predicts log E[num_pfvs] from (kappa, c2, H, h11). Trained with Poisson NLL."""

    def __init__(
        self,
        max_h11: int,
        d_enc: int = 128,
        d_h11: int = 16,
        d_head: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kappa = KappaEncoder(max_h11=max_h11, d_out=d_enc)
        self.c2    = C2Encoder   (max_h11=max_h11, d_out=d_enc)
        self.H     = HEncoder    (max_h11=max_h11, d_out=d_enc)
        self.h11_emb = nn.Embedding(max_h11 + 1, d_h11)

        self.trunk = nn.Sequential(
            nn.Linear(3 * d_enc + d_h11, d_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_head, d_head),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.count_head  = nn.Linear(d_head, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        k = self.kappa(batch['kappa_idx'], batch['kappa_v'], batch['kappa_mask'])
        c = self.c2   (batch['c2'],         batch['c2_mask'])
        h = self.H    (batch['H'],          batch['H_row_mask'], batch['H_col_mask'])
        e = self.h11_emb(batch['h11'])
        z = self.trunk(torch.cat([k, c, h, e], dim=-1))
        return self.count_head(z).squeeze(-1)   # log_lambda  (B,)
