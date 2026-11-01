"""How many perturbations show a clear response in a line? Pseudobulk z-scores against controls
(Poisson noise on summed counts, inflated x4 for overdispersion), counted over genes >= 5 CPM."""
import sys
import numpy as np, pandas as pd
from vcc.stats import load

vcc = set(pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").target_gene)


def n_responsive_genes(st, z_thr=5.0, overdisp=4.0, min_cells=30):
    names = list(st["names"]); S = st["sums"]; c = names.index("__ctrl__")
    tot = S.sum(1, keepdims=True); cpm = S / tot * 1e6; expr = cpm[c] >= 5
    out = {}
    for i, n in enumerate(names):
        if n == "__ctrl__" or st["ncell"][i] < min_cells:
            continue
        lfc = np.log((S[i] + 1) / tot[i] * tot[c] / (S[c] + 1))
        se = np.sqrt(overdisp * (1 / (S[i] + 1) + 1 / (S[c] + 1)))
        out[n] = int(((np.abs(lfc / se) > z_thr) & expr & st["measured"]).sum())
    return pd.Series(out)


if __name__ == "__main__":
    for line in sys.argv[1:]:
        s = n_responsive_genes(load(line)); isv = s.index.isin(list(vcc))
        for k in (10, 30, 100):
            print(f"{line}: >= {k:3d} responsive genes: VCC targets {int((s[isv] >= k).sum())}/{isv.sum()}, "
                  f"aux {int((s[~isv] >= k).sum())}/{(~isv).sum()}")
        s.to_csv(f"/home/zxc29051617/VCC/data/stage3/stats/{line}_responsive.csv")
