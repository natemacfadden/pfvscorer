"""Train the PFV count model (Poisson NLL on log_lambda).

Polytope-level train/val/test split. Reports median rel err and AUC at >100, >1000.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .dataset import DEFAULT_PARQUET, ConiDataset, collate
from .model import PFVCountModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default=DEFAULT_PARQUET)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--max_h11', type=int, default=10)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--test_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--h11_filter', type=int, nargs='*', default=None,
                    help='restrict to specific h11 values (default: all)')
    ap.add_argument('--load_ckpt', default=None,
                    help='path to checkpoint; if given, skip training and eval only')
    ap.add_argument('--ckpt_out', default='checkpoint.pt')
    ap.add_argument('--augment', action='store_true',
                    help='enable GLSM-basis (unimodular) augmentation for training')
    ap.add_argument('--aug_n_ops', type=int, default=4)
    ap.add_argument('--aug_k_range', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def predict(model, loader, device):
    """One forward pass. Returns (preds, targets, h11s) as np arrays."""
    model.eval()
    preds, targets, h11s = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            log_lambda = model(batch)
            preds.append(torch.exp(log_lambda).cpu().numpy())
            targets.append(batch['num_pfvs'].cpu().numpy())
            h11s.append(batch['h11'].cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(h11s)


def auc(scores, y_true):
    """Mann-Whitney U: P(score_pos > score_neg) for a random (pos, neg) pair."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(preds, targets):
    abs_errs = np.abs(preds - targets)
    nz = targets > 0
    rel = abs_errs[nz] / targets[nz]
    return {
        'med_abs':  float(np.median(abs_errs)),
        'rel_med':  float(np.median(rel)) if nz.any() else float('nan'),
        'rel_mean': float(rel.mean())     if nz.any() else float('nan'),
        'auc_100':  auc(preds, targets > 100),
        'auc_1000': auc(preds, targets > 1000),
    }


def polytope_split(df, val_frac, test_frac, seed):
    pairs = df[['h11', 'polyID']].drop_duplicates().to_records(index=False).tolist()
    pairs = [(int(h), int(p)) for (h, p) in pairs]
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_val  = int(val_frac  * len(pairs))
    n_test = int(test_frac * len(pairs))
    val_set  = set(pairs[:n_val])
    test_set = set(pairs[n_val:n_val + n_test])
    keys = list(zip(df['h11'].astype(int), df['polyID'].astype(int)))
    train_idx, val_idx, test_idx = [], [], []
    for i, k in enumerate(keys):
        if k in val_set:    val_idx.append(i)
        elif k in test_set: test_idx.append(i)
        else:               train_idx.append(i)
    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_full = ConiDataset(args.parquet, h11_filter=args.h11_filter)

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location='cpu', weights_only=False)
        train_idx = list(ckpt['train_idx'])
        val_idx   = list(ckpt['val_idx'])
        test_idx  = list(ckpt['test_idx'])
        print(f"loaded splits from {args.load_ckpt}")
    else:
        train_idx, val_idx, test_idx = polytope_split(
            ds_full.df, args.val_frac, args.test_frac, args.seed)

    if args.augment:
        ds_train_src = ConiDataset(
            args.parquet, h11_filter=args.h11_filter, augment=True,
            aug_n_ops=args.aug_n_ops, aug_k_range=args.aug_k_range,
        )
    else:
        ds_train_src = ds_full
    ds_train = Subset(ds_train_src, train_idx)
    ds_val   = Subset(ds_full,      val_idx)
    ds_test  = Subset(ds_full,      test_idx)
    print(f"split (polytope-level): train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,  collate_fn=collate,
                          num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, collate_fn=collate)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device(args.device)
    model = PFVCountModel(max_h11=args.max_h11).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.load_ckpt:
        model.load_state_dict(ckpt['model_state'])
        print(f"loaded model weights from {args.load_ckpt}")

    poisson_nll = nn.PoissonNLLLoss(log_input=True)

    if not args.load_ckpt:
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            running, n_seen = 0.0, 0
            for batch in dl_train:
                batch = to_device(batch, device)
                y = batch['num_pfvs'].float()
                log_lambda = model(batch)
                loss = poisson_nll(log_lambda, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running += loss.item() * y.shape[0]
                n_seen  += y.shape[0]
            sched.step()
            train_loss = running / n_seen
            val_preds, val_targets, _ = predict(model, dl_val, device)
            m = metrics(val_preds, val_targets)
            print(f"  ep {ep:3d}  train_loss={train_loss:7.3f}  "
                  f"val_rel_med={m['rel_med']:.3f}  val_AUC>1000={m['auc_1000']:.3f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  t={time.time()-t0:6.1f}s")
        torch.save({
            'model_state': model.state_dict(),
            'train_idx':   train_idx,
            'val_idx':     val_idx,
            'test_idx':    test_idx,
            'args':        vars(args),
        }, args.ckpt_out)
        print(f"saved checkpoint -> {args.ckpt_out}")

    # final detailed eval on val and test
    print()
    for name, dl in [('val', dl_val), ('test', dl_test)]:
        preds, targets, h11s = predict(model, dl, device)
        m = metrics(preds, targets)
        nz = targets > 0
        print(f"=== {name} ===")
        print(f"  n             = {len(preds)}  (target>0: {int(nz.sum())})")
        print(f"  median |delta|    = {m['med_abs']:7.2f}    (counts)")
        print(f"  median |delta|/y  = {m['rel_med']:.3f}    (rel err on nonzero)")
        print(f"  mean   |delta|/y  = {m['rel_mean']:.3f}")
        print(f"  AUC > 100     = {m['auc_100']:.3f}    (positives: {int((targets>100).sum())})")
        print(f"  AUC > 1000    = {m['auc_1000']:.3f}    (positives: {int((targets>1000).sum())})")


if __name__ == '__main__':
    main()
