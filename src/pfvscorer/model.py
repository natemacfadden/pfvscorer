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
# Description:  Geometry -> per-threshold presence classifier (PFV richness).
# -----------------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import KappaEncoder, C2Encoder, HEncoder, signed_log1p


class PFVRichnessModel(nn.Module):
    """Presence classifier over (kappa, c2, H, h11, h21, B, dilation).

    Three geometry encoders + an h11 embedding + a small MLP over the scalars
    [h21, B, dilation] feed a shared trunk; the head emits one logit per count
    threshold. probs() applies a sigmoid -> P(#PFVs in the (B, dil) window >
    threshold). n_out > 1 gives several threshold heads (e.g. >0 and >50) from
    one shared encoder.
    """

    def __init__(
        self,
        max_h11: int,
        d_enc: int = 128,
        d_h11: int = 16,
        d_scalar: int = 32,
        d_head: int = 256,
        dropout: float = 0.1,
        n_out: int = 1,
    ):
        super().__init__()
        self.n_out = n_out      # one logit per count threshold
        self.kappa = KappaEncoder(max_h11=max_h11, d_out=d_enc)
        self.c2    = C2Encoder   (max_h11=max_h11, d_out=d_enc)
        self.H     = HEncoder    (max_h11=max_h11, d_out=d_enc)
        self.h11_emb = nn.Embedding(max_h11 + 1, d_h11)

        # scalar conditioning: signed_log1p of [h21, B, dil]
        self.scalar_mlp = nn.Sequential(
            nn.Linear(3, d_scalar),
            nn.GELU(),
            nn.Linear(d_scalar, d_scalar),
        )

        self.trunk = nn.Sequential(
            nn.Linear(3 * d_enc + d_h11 + d_scalar, d_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_head, d_head),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(d_head, n_out)   # one logit per threshold

    def forward(self, batch: dict) -> torch.Tensor:
        """Raw logits per threshold head.

        Parameters
        ----------
        batch : dict
            Collated batch produced by dataset.collate.

        Returns
        -------
        torch.Tensor
            Logits of shape (N,) if n_out == 1, else (N, n_out).
        """
        k = self.kappa(batch['kappa_idx'], batch['kappa_v'], batch['kappa_mask'])
        c = self.c2   (batch['c2'],         batch['c2_mask'])
        h = self.H    (batch['H'],          batch['H_row_mask'], batch['H_col_mask'])
        e = self.h11_emb(batch['h11'])
        scalars = torch.stack([batch['h21'], batch['B'], batch['dil']], dim=-1)
        s = self.scalar_mlp(signed_log1p(scalars))
        z = self.trunk(torch.cat([k, c, h, e, s], dim=-1))
        return self.head(z).squeeze(-1)        # (N,) if n_out==1 else (N, n_out)

    def probs(self, batch: dict) -> torch.Tensor:
        """Sigmoid of forward() -- P(count > threshold) per head, same shape."""
        return torch.sigmoid(self.forward(batch))
