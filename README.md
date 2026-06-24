# pfvscorer

*Nate MacFadden, Liam McAllister Group, Cornell*

A neural **richness classifier** for the classes of Diophantine problems whose
solutions are perturbatively flat vacua (PFVs) in string theory. The geometry of
a conifold defines such a class (parameterized by `p in Z^h11`); given that
geometry, the model predicts the probability that the class contains **more than
a threshold number of solutions** (e.g. `>0` and `>50`), so that a downstream
search ([pfvs](https://github.com/natemacfadden/pfvs)) can be allocated in
proportion to how rich each geometry is likely to be.

## Why a scorer

[Certain string-theory problems](https://arxiv.org/abs/2406.13751) largely reduce
to finding PFVs. The [pfvs](https://github.com/natemacfadden/pfvs) kernel
enumerates solutions quickly, but the problem family is parameterized by
`p in Z^h11`, so a naive search is unbounded. `pfvscorer` predicts, before any
expensive enumeration, how many solutions a geometry is likely to yield --
turning "search everything" into "search where solutions are likely, as deeply
as they are likely to be there."

## Inputs and the symmetry of each

A conifold supplies several geometric objects plus scalar context. Each is
encoded by a network that respects *that object's* symmetry, rather than
flattening everything into a single sequence:

| input | object | encoder | symmetry respected |
|---|---|---|---|
| `kappa` | triple-intersection tensor `Z^(h11 x h11 x h11)` | Set Transformer over symmetric-COO tokens; within a token the three indices share an embedding and are summed | fully symmetric in its three indices (a Deep Sets aggregation within each token); tokens form an unordered set (permutation-invariant pooling) |
| `c2` | second Chern class `Z^h11` | per-position embedding + small Transformer + masked mean-pool | positions are *ordered* (one per basis divisor) |
| `H` | Mori-cone hyperplane normals `Z^(N x (h11-1))` | Deep Sets over rows, position-aware within each row | rows are an unordered *set*; columns are ordered |
| `h11` | scalar | learned embedding | -- |
| `h21, B, dilation` | scalar context | `signed_log1p` + small MLP | -- |

The encodings feed a shared MLP trunk that emits one sigmoid logit per count
threshold; `probs()` returns `P(#PFVs > threshold)` for each head.

## Status

Research code. A trained model and a calibration experiment live under
`experiments`.

## Installation

```
pip install -e .
```

Requires PyTorch. Optional extras:

```
pip install -e ".[test]"          # pytest
pip install -e ".[experiments]"   # scikit-learn, for the calibration experiment
```

## Data

The full dataset (18,253 conifolds, ~17 MB) and the trained checkpoint behind the
`experiments/` metrics are hosted on Hugging Face:
[natemacfadden/calabi-yau-coni-pfvs](https://huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs).
A small stratified sample (`data/sample_coni_pfvs.parquet`, 50 rows) ships with the
repo, so the test suite and a quick pipeline run work from a fresh clone with no download.

To reproduce the `experiments/` metrics, fetch the full data and checkpoint into the
paths the scripts expect (`huggingface-cli` comes from `pip install huggingface_hub`):

```bash
huggingface-cli download natemacfadden/calabi-yau-coni-pfvs coni_pfvs.parquet \
    --repo-type dataset --local-dir data
huggingface-cli download natemacfadden/calabi-yau-coni-pfvs coni_pfvs_bce2.pt \
    --repo-type dataset --local-dir checkpoints
```

See `data/data_description.md` for the schema.
