# coni_pfvs.parquet

Number of PFVs found for various conifolds, found via [pfvs](https://github.com/natemacfadden/pfvs).

This file holds both [the data necessary to run the `pfv` code](https://github.com/natemacfadden/pfvs/blob/53d99a09ad6c85e23a49d509ff08e23b82fc82a4/pfvs/cydata.py#L33) (intersection numbers `kappa_coo`, second chern class `c2`, hyperplanes `H`) and the resultant PFVs (`M`, `K`).

The PFV runs have configuration, primarily in the number of `p`-vectors studied and the maximum dilation considered. These are parameterized PFV-by-PFV using `pfv_infnorm` and `pfv_reqdil` respectively. There is also larger-scale information like the number of PFVs found `npfvs`, the number of `p`-vectors studied `frontier_npvecs`, how large of a box of PFVs this corresponds to `frontier_infnorm`, and the dilation used for the PFV searches `frontier_dil`.

Order is

h11, h21, polyID, classID, coniID, # coni specification
kappa_coo, c2, H,                  # coni relevant data
M, K, pfv_infnorm, pfv_reqdil,     # PFV-level data
npfvs, frontier_npvecs, frontier_infnorm, frontier_dil # search-level data
