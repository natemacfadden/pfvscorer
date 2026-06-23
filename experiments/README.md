# pfvscorer experiments

A trustable **pre-filter**: from a conifold's geometry, predict whether a search
to a given `(B, dilation)` window will find PFVs -- to triage which conifolds are
worth an expensive deep search. Not a count predictor. Data: `data/coni_pfvs.parquet`
(one row per conifold; see `data/coni_pfvs.md`). All numbers below are measured on
the held-out splits; nothing here is inferred.

## Split (coni-level, seed 0, 0.70 / 0.15 / 0.15)
train = 12,779   val = 2,737   test = 2,737
- train: fits weights.  val: model/threshold selection + calibration fit.  test: reported once.

## Model: two-head presence classifier
`src/pfvscorer/` -- 3 geometry encoders (`kappa_coo`, `c2`, `H`) + `h11` embedding +
a scalar MLP over `[h21, B, dilation]` -> shared trunk -> one sigmoid logit per count
threshold. `(B, dilation)` is the chosen search depth (conditioning, not leakage); the
training label is `count_in_window > threshold`, BCE per head. Two heads: `>0` (any) and
`>50` (rich).

Train:  `python -m pfvscorer.train --augment --bce_thresh 0 50 --ckpt_out checkpoints/coni_pfvs_bce2.pt`
(`--augment` = signed-permutation GLSM-basis augmentation, which preserves the L-inf box
and the ellipsoid form, so the (B,dil) labels stay exact.)

Checkpoint: `checkpoints/coni_pfvs_bce2.pt`; isotonic calibrators (fit on val):
`checkpoints/coni_pfvs_bce2_calib.pkl`.

## Results (test, scored at each coni's deepest frontier)
| head        | AUC   | ECE raw -> calibrated |
|-------------|-------|-----------------------|
| `>0`  (any) | 0.815 | 0.151 -> 0.023        |
| `>50` (rich)| 0.843 | 0.055 -> 0.018        |

Calibrated operating points (test): `P(>0)>0.7` -> precision 84% / recall 72%;
`P(>0)>0.9` -> precision 93%. `P(>50)>0.25` -> flags ~27% of all conis, captures ~74%
of the rich (>50) ones (~39% of the flagged are truly >50).

## Files
- `calibrate.py`   isotonic calibration (fit val, eval test); writes calibration.png
                   (per-head P, bin rate +/- 1 sigma over the 0/1 scatter; BINS=30)
- `filter_eval.py` presence AUC + recall-vs-budget; `--head {0,1} --label_thresh {0,50}`

## Validations (recorded)
- Not riding B: raw `frontier_B` alone is AUC 0.47 (chance) for `npfvs>0`; positives and
  nulls have identical median `frontier_B` (13); the geometry carries the signal.
- Negative probe: a genuine null (searched to B=170, npfvs=0) stays P(>0) <= 0.08 as B,dil
  are swept to (800, 400) -- big B does not flip it positive.
- Positive probe: P(>0) is ~0 below a coni's onset (min infnorm) and rises as the true
  count turns on -- B is used sensibly.
- Count *ranking* (abandoned) collapses at h11=10,11 (Spearman ~0.3-0.4), and a dedicated
  high-h11 model did no better -> a representation limit, not capacity. Presence/threshold
  is the tractable task and is why the filter works there (AUC ~0.8 at all h11).

## Caveats
- "null" = no PFVs in the SEARCHED box; the deep search was incomplete (worst at high
  h11), so a null could have PFVs beyond its frontier.
- Best used as a prioritizer (trust the top of the ranking). `P(>50)` ranks the rich
  conis well but, being calibrated, tops out ~0.5 -- it is a recall-oriented prioritizer,
  not a high-precision detector.
