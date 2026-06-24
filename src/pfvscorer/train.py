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
# Description:  Train the PFV presence classifier (multi-threshold BCE).
#               Each example conditions on a sampled (B', dil') window; each
#               head predicts P(#PFVs in window > its threshold). Reports
#               per-head AUC on val/test.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .dataset import DEFAULT_PARQUET, ConiDataset, collate
from .model import PFVRichnessModel


def parse_args():
    """Parse command-line training arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_h11", type=int, default=11)
    ap.add_argument(
        "--bce_thresh",
        type=float,
        nargs="+",
        default=[0.0, 50.0],
        help="presence head(s): one per count threshold (default >0 and >50)",
    )
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--h11_filter",
        type=int,
        nargs="*",
        default=None,
        help="restrict to specific h11 values (default: all)",
    )
    ap.add_argument("--ckpt_out", default="checkpoint.pt")
    ap.add_argument(
        "--augment",
        action="store_true",
        help="enable signed-permutation GLSM-basis augmentation for training",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()


def to_device(batch: dict, device) -> dict:
    """Move every tensor in a batch dict to ``device``."""
    return {k: v.to(device) for k, v in batch.items()}


def bce_loss(logits: torch.Tensor, y: torch.Tensor, thresholds) -> torch.Tensor:
    """Mean BCE over presence heads; head j's label is (y > thresholds[j])."""
    tgt = torch.stack([(y > t).float() for t in thresholds], dim=-1)  # (N, H)
    lg = logits if logits.dim() == 2 else logits.unsqueeze(-1)
    return nn.functional.binary_cross_entropy_with_logits(lg, tgt)


def predict(model, loader, device):
    """Run the model over a loader.

    Returns
    -------
    tuple of np.ndarray
        (probs, counts): per-head probabilities of shape (N, H) and the
        conditioned counts of shape (N,).
    """
    model.eval()
    P, C = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            p = model.probs(batch)
            if p.dim() == 1:
                p = p.unsqueeze(-1)
            P.append(p.cpu().numpy())
            C.append(batch["count"].cpu().numpy())
    return np.concatenate(P), np.concatenate(C)


def auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """ROC-AUC = P(a random positive scores above a random negative).

    Returns nan if either class is empty.
    """
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-sum identity for AUC; threshold-free (no tie correction)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def head_aucs(P: np.ndarray, C: np.ndarray, thresholds) -> list:
    """AUC of each head's prob (P[:, j]) vs the label (C > thresholds[j])."""
    return [auc(P[:, j], C > t) for j, t in enumerate(thresholds)]


def coni_split(n_rows: int, val_frac: float, test_frac: float, seed: int):
    """Shuffled row split into (train, val, test) index lists.

    Each row is one conifold, so a row split is a coni-level split with no
    leakage (augmentation happens inside __getitem__).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_val = int(val_frac * n_rows)
    n_test = int(test_frac * n_rows)
    val_idx = idx[:n_val].tolist()
    test_idx = idx[n_val : n_val + n_test].tolist()
    train_idx = idx[n_val + n_test :].tolist()
    return train_idx, val_idx, test_idx


def main():
    """Train the classifier, report per-head AUC, and save the checkpoint."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_eval = ConiDataset(
        args.parquet, h11_filter=args.h11_filter, augment=False, train=False
    )
    train_idx, val_idx, test_idx = coni_split(
        len(ds_eval), args.val_frac, args.test_frac, args.seed
    )

    ds_train_src = ConiDataset(
        args.parquet, h11_filter=args.h11_filter, augment=args.augment, train=True
    )
    ds_train = Subset(ds_train_src, train_idx)
    ds_val = Subset(ds_eval, val_idx)
    ds_test = Subset(ds_eval, test_idx)
    print(
        f"split (coni-level): train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}"
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch, shuffle=False, collate_fn=collate
    )
    dl_test = DataLoader(
        ds_test, batch_size=args.batch, shuffle=False, collate_fn=collate
    )

    device = torch.device(args.device)
    model = PFVRichnessModel(max_h11=args.max_h11, n_out=len(args.bce_thresh)).to(
        device
    )
    print(
        f"model params: {sum(p.numel() for p in model.parameters()):,}  "
        f"heads (count> ): {[int(t) for t in args.bce_thresh]}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for batch in dl_train:
            batch = to_device(batch, device)
            y = batch["count"].float()
            loss = bce_loss(model(batch), y, args.bce_thresh)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * y.shape[0]
            n_seen += y.shape[0]
        sched.step()
        Pv, Cv = predict(model, dl_val, device)
        aucs = head_aucs(Pv, Cv, args.bce_thresh)
        auc_str = "  ".join(f">{int(t)}:{a:.3f}" for t, a in zip(args.bce_thresh, aucs))
        print(
            f"  ep {ep:3d}  train_bce={running / n_seen:6.3f}  val AUC {auc_str}  "
            f"lr={sched.get_last_lr()[0]:.2e}  t={time.time() - t0:6.1f}s"
        )
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        },
        args.ckpt_out,
    )
    print(f"saved checkpoint -> {args.ckpt_out}")

    print()
    for name, dl in [("val", dl_val), ("test", dl_test)]:
        P, C = predict(model, dl, device)
        aucs = head_aucs(P, C, args.bce_thresh)
        print(f"=== {name} (n={len(C)}) ===")
        for t, a in zip(args.bce_thresh, aucs):
            print(
                f"  AUC(count>{int(t)}) = {a:.3f}   (positives: {int((C > t).sum())})"
            )


if __name__ == "__main__":
    main()
