"""Is the perturbation effect energy concentrated in a few directions? SVD of the (targets x genes) effect
matrix of a line: share of total squared effect captured by the top-k components, after removing the
across-target mean (the 'generic response' the mse baseline already gets for free)."""
import json, sys
import numpy as np, pandas as pd
from vcc.effects import effect_matrix
from vcc.stats import load
line = sys.argv[1]; minresp = int(sys.argv[2]) if len(sys.argv) > 2 else 0
idx, E, c, meas = effect_matrix(line)
expr = (2 ** c - 1 >= 5) & meas
tg = list(idx)
if minresp:
    from vcc.responsive import n_responsive_genes
    r = n_responsive_genes(load(line)); tg = [t for t in tg if r.get(t, 0) >= minresp]
X = np.nan_to_num(E[idx.get_indexer(tg)][:, expr])
tot = (X ** 2).sum(); mean_part = (X.mean(0, keepdims=True) ** 2).sum() * len(X)
Xc = X - X.mean(0, keepdims=True)
sv = np.linalg.svd(Xc, compute_uv=False) ** 2
print(f"{line}: {len(tg)} targets (>= {minresp} responsive genes), {expr.sum()} expressed genes")
print(f"  total effect energy: generic (mean response) {mean_part/tot:.1%}, target-specific {1-mean_part/tot:.1%}")
cum = np.cumsum(sv) / sv.sum()
print("  target-specific energy in the top-k components: " + "  ".join(f"k={k}: {cum[k-1]:.1%}" for k in (1, 2, 3, 5, 10, 20) if k <= len(cum)))
