# pfvscorer
*Nate MacFadden, Liam McAllister Group, Cornell*

A transformer for predicting the number of solutions that certain classes of Diophantine problems each have.

This is designed for string theory purposes, for which the solutions are **perturbatively flat vacua (PFVs)** and the classes of problems are those defined by different **conifolds**.

## Problem Statement

See [pfvs](https://github.com/natemacfadden/pfvs).

## Description

A given conifold defines certain data:
1) a scalar $h^{1,1}$,
2) (triple intersection numbers) a tensor $\kappa\in\mathbb{Z}^{h^{1,1},h^{1,1},h^{1,1}}$,
3) (second chern class) a vector $c_2 \in \mathbb{Z}^{h^{1,1}}$, and
4) (hyperplane normals) a matrix $H \in \mathbb{Z}^{N,h^{1,1}-1}$.

This data defines a class (parameterized by a parameter $p\in\mathbb{Z}^{h^{1,1}}$) of Diophantine problems. (Note for the physicists: this data assumes a basis where the conifold curve is of the form $(1, 0, \dots, 0)$; a change-of-basis should be applied if necessary).

This model encodes the inputs $\kappa$, $c_2$, and $H$ via three separate encoder networks; $h^{1,1}$ enters through a learned embedding.. These encodings are then passed through a trunk decoder network that is trained (using Poisson NLL loss) to predict the log count of solutions.

| input    | encoder        | symmetry it respects                                         |
|----------|----------------|-------------------------------------------------------------|
| $\kappa$ | Set Transformer over symmetric-COO tokens | tokens are an unordered set; the index triple is symmetric  |
| $c_2$    | small Transformer over `(position, value)` | positions are ordered (one per basis divisor)               |
| $H$      | Deep Sets over rows | rows are an unordered set; columns are ordered              |

### Intended use

[Certain string theory problems](https://arxiv.org/abs/2406.13751) largely reduce to finding PFVs with certain properties. My kernel [pfvs](https://github.com/natemacfadden/pfvs) is able to quickly enumerate solutions, but the parameterization of the class of problems via $p\in\mathbb{Z}^{h^{1,1}}$ makes this problem naively unbounded. Before sampling $p$, the scorer model in this repo predicts how rich the geometry (conifold) is, guiding how many $p$ vectors to search and how deeply to search them (there are other parameters in the kernel).

## Installation

```
pip install -e .
```

Requires PyTorch. Optional extras: `pip install -e ".[test]"` (pytest),
`pip install -e ".[experiments]"` (scikit-learn, for the calibration experiment).
