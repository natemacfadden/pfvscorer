"""Calibrate the two-head BCE filter: isotonic map fit on val, evaluated on test.
Reports Brier/ECE before vs after, a reliability diagram, and operating points.
Saves the fitted calibrators next to the checkpoint."""

from __future__ import annotations
import os
import pickle
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from scipy.stats import beta as _Beta

from pfvscorer.dataset import DEFAULT_PARQUET, collate
from pfvscorer.model import PFVRichnessModel
from pfvscorer.train import coni_split, auc

CKPT = "checkpoints/coni_pfvs_bce2.pt"
BINS = 30


def bs(r, B, D):
    kc = [list(x) for x in r.kappa_coo]
    if kc:
        arr = torch.tensor(kc, dtype=torch.long)
        ki, kv = arr[:, :3].contiguous(), arr[:, 3].float()
    else:
        ki = torch.zeros((0, 3), dtype=torch.long)
        kv = torch.zeros((0,))
    H = [list(x) for x in r.H]
    Ht = (
        torch.tensor(H, dtype=torch.long)
        if H
        else torch.zeros((0, 0), dtype=torch.long)
    )
    return {
        "kappa_idx": ki,
        "kappa_v": kv,
        "c2": torch.tensor(list(r.c2), dtype=torch.long),
        "H": Ht,
        "h11": torch.tensor(int(r.h11)),
        "h21": torch.tensor(float(r.h21)),
        "B": torch.tensor(float(B)),
        "dil": torch.tensor(float(D)),
        "count": torch.tensor(0),
    }


def score(model, df, idxs, dev):
    """Raw head probabilities at each coni's deepest frontier -> (N,2), plus npfvs."""
    samp, npf = [], []
    for i in idxs:
        r = df.iloc[i]
        fb = np.asarray(r.frontier_infnorm)
        fd = np.asarray(r.frontier_dil)
        k = int(np.argmax(fb * fd))
        samp.append(bs(r, int(fb[k]), int(fd[k])))
        npf.append(int(r.npfvs))
    out = []
    for j in range(0, len(samp), 512):
        b = {k: v.to(dev) for k, v in collate(samp[j : j + 512]).items()}
        with torch.no_grad():
            out.append(torch.sigmoid(model(b)).cpu().numpy())
    return np.concatenate(out), np.array(npf)


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def binstats(p, y, bins=BINS):
    """Per-bin: mean predicted P, Beta-posterior mean rate, posterior std (1 sigma)."""
    edges = np.linspace(0, 1, bins + 1)
    xs, mean, std = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n < 1:
            continue
        k = int(y[m].sum())
        xs.append(p[m].mean())
        mean.append((k + 1) / (n + 2))
        std.append(float(_Beta.std(k + 1, n - k + 1)))
    return np.array(xs), np.array(mean), np.array(std)


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ckpt records the absolute parquet path from training; fall back to the local
    # default (e.g. fetched from hf into data/) when that path isn't present here
    parquet = (
        a["parquet"]
        if a.get("parquet") and os.path.exists(a["parquet"])
        else DEFAULT_PARQUET
    )
    df = pq.read_table(parquet).to_pandas()
    # use the split stored in the ckpt; recompute only for an older ckpt without it
    if "val_idx" in ckpt and "test_idx" in ckpt:
        val, test = ckpt["val_idx"], ckpt["test_idx"]
    else:
        _, val, test = coni_split(len(df), a["val_frac"], a["test_frac"], a["seed"])
    model = PFVRichnessModel(max_h11=a["max_h11"], n_out=len(a["bce_thresh"])).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    Pv, npv = score(model, df, val, dev)
    Pt, npt = score(model, df, test, dev)
    heads = [(0, 0, "P(>0)"), (1, 50, "P(>50)")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    calibrators = {}
    for col, thr, name in heads:
        yv = (npv > thr).astype(float)
        yt = (npt > thr).astype(float)
        raw_v, raw_t = Pv[:, col], Pt[:, col]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_v, yv)
        cal_t = iso.predict(raw_t)
        calibrators[thr] = iso
        print(f"\n=== head {col}  {name}  (test, base rate {yt.mean() * 100:.0f}%) ===")
        print(f"  AUC (unchanged): {auc(raw_t, yt.astype(bool)):.3f}")
        print(
            f"  Brier  raw={np.mean((raw_t - yt) ** 2):.3f} -> cal={np.mean((cal_t - yt) ** 2):.3f}"
        )
        print(f"  ECE    raw={ece(raw_t, yt):.3f} -> cal={ece(cal_t, yt):.3f}")
        print(
            f"  raw P range [{raw_t.min():.2f},{raw_t.max():.2f}]  cal P range [{cal_t.min():.2f},{cal_t.max():.2f}]"
        )
        print(f"  operating points (calibrated, test):")
        for tau in (0.25, 0.3, 0.5, 0.7, 0.9):
            fl = cal_t > tau
            if fl.sum():
                prec = yt[fl].mean()
                rec = yt[fl].sum() / max(yt.sum(), 1)
                print(
                    f"    P>{tau}: flag {fl.mean() * 100:4.0f}%  precision {prec * 100:4.0f}%  recall {rec * 100:4.0f}%"
                )
        rng = np.random.default_rng(0)
        xb, mb, sb = binstats(cal_t, yt)
        ax[col].scatter(
            cal_t,
            yt + rng.uniform(-0.03, 0.03, len(yt)),
            s=5,
            alpha=0.05,
            color="k",
            zorder=1,
        )
        ax[col].plot([0, 1], [0, 1], "k:", lw=1, zorder=2, label="perfect")
        ax[col].errorbar(
            xb,
            mb,
            yerr=sb,
            fmt="o",
            ms=5,
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.5,
            capsize=3,
            zorder=4,
            label="bin rate",
        )
        ax[col].set_title(f"head {col}: {name} (test data)")
        ax[col].set_xlabel(f"model prob for >{thr} PFVs")
        ax[col].set_ylabel(f"P(>{thr} PFVs | model prediction)")
        ax[col].set_xlim(-0.04, 1.04)
        ax[col].set_ylim(-0.1, 1.12)
        ax[col].legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
    fig.tight_layout()
    fig.savefig("experiments/calibration.png", dpi=120)
    pickle.dump(calibrators, open("checkpoints/coni_pfvs_bce2_calib.pkl", "wb"))
    print(
        "\nsaved -> experiments/calibration.png, checkpoints/coni_pfvs_bce2_calib.pkl"
    )


if __name__ == "__main__":
    main()
