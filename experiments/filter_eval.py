"""Evaluate a count model as an 'any PFVs' pre-filter: score = predicted count
at the (deepest) frontier, label = (npfvs > 0). Reports ROC-AUC overall and per
h11, plus a recall-vs-budget table (study the top X%, catch Y% of positives)."""
from __future__ import annotations
import argparse
import numpy as np
import torch
import pyarrow.parquet as pq

from pfvscorer.dataset import DEFAULT_PARQUET, collate
from pfvscorer.model import PFVRichnessModel
from pfvscorer.train import coni_split, auc


def build_sample(row, Bp, Dp):
    kc = [list(x) for x in row.kappa_coo]
    if kc:
        arr = torch.tensor(kc, dtype=torch.long); ki, kv = arr[:, :3].contiguous(), arr[:, 3].float()
    else:
        ki = torch.zeros((0, 3), dtype=torch.long); kv = torch.zeros((0,))
    H = [list(x) for x in row.H]
    Ht = torch.tensor(H, dtype=torch.long) if H else torch.zeros((0, 0), dtype=torch.long)
    return {'kappa_idx': ki, 'kappa_v': kv, 'c2': torch.tensor(list(row.c2), dtype=torch.long),
            'H': Ht, 'h11': torch.tensor(int(row.h11), dtype=torch.long),
            'h21': torch.tensor(float(row.h21), dtype=torch.float32),
            'B': torch.tensor(float(Bp)), 'dil': torch.tensor(float(Dp)),
            'count': torch.tensor(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/coni_pfvs_bce2.pt')
    ap.add_argument('--label_thresh', type=float, default=0.0, help='positive = npfvs > this')
    ap.add_argument('--head', type=int, default=0, help='which head to score (multi-head models)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = ckpt['args']; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pq.read_table(a.get('parquet', DEFAULT_PARQUET)).to_pandas()
    if a.get('h11_filter'):
        df = df[df.h11.isin(a['h11_filter'])].reset_index(drop=True)
    val_idx = ckpt['val_idx'] if 'val_idx' in ckpt else coni_split(len(df), a['val_frac'], a['test_frac'], a['seed'])[1]

    n_out = len(a.get('bce_thresh', [0.0]))
    model = PFVRichnessModel(max_h11=a['max_h11'], n_out=n_out).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval()

    # score each val coni at its deepest frontier (largest box area)
    samples, h11s, labels = [], [], []
    for idx in val_idx:
        r = df.iloc[idx]
        fb = np.asarray(r.frontier_infnorm, np.int64); fd = np.asarray(r.frontier_dil, np.int64)
        k = int(np.argmax(fb * fd))
        samples.append(build_sample(r, int(fb[k]), int(fd[k])))
        h11s.append(int(r.h11)); labels.append(int(r.npfvs > args.label_thresh))
    scores = []
    for i in range(0, len(samples), 512):
        batch = {k: v.to(device) for k, v in collate(samples[i:i + 512]).items()}
        with torch.no_grad():
            e = model.probs(batch)
            if e.dim() > 1: e = e[:, args.head]
            scores.append(e.cpu().numpy())
    scores = np.concatenate(scores); h11s = np.array(h11s); labels = np.array(labels).astype(bool)

    print(f"ckpt={args.ckpt}  head={args.head}  label=npfvs>{args.label_thresh:g}  "
          f"val conis={len(scores)}  positives={labels.sum()} ({labels.mean()*100:.0f}%)")
    print(f"  AUC overall = {auc(scores, labels):.3f}")
    for hv in np.unique(h11s):
        m = h11s == hv
        print(f"  h11={hv:>2}: AUC={auc(scores[m], labels[m]):.3f}  "
              f"(n={m.sum()}, pos={labels[m].sum()} [{labels[m].mean()*100:.0f}%])")

    # recall-vs-budget (overall): study top X% by score, catch Y% of positives
    order = np.argsort(-scores); lab_sorted = labels[order]; P = labels.sum()
    print("  study top X%  ->  recall of positives | nulls skipped")
    for frac in (0.1, 0.25, 0.5, 0.75):
        k = int(frac * len(scores))
        rec = lab_sorted[:k].sum() / P
        nulls_total = (~labels).sum(); nulls_skipped = (~lab_sorted[k:]).sum() if k < len(scores) else nulls_total
        print(f"    top {int(frac*100):>2}%:  recall={rec*100:5.1f}%   nulls correctly skipped={nulls_skipped/nulls_total*100:5.1f}%")
    # budget to catch 90/95% of positives
    cum = np.cumsum(lab_sorted) / P
    for target in (0.9, 0.95, 0.99):
        need = (np.searchsorted(cum, target) + 1) / len(scores)
        print(f"    to catch {int(target*100)}% of positives: study {need*100:.1f}% of conis")


if __name__ == '__main__':
    main()
