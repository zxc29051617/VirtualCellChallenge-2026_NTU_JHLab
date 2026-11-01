"""How much do the sources agree on the VCC targets? Pairwise, over targets both sources measured:
mean per-target Pearson of the effect vectors (log2 fold change on genes both measure and both controls
express >= 5 CPM), computed on RAW effects and after removing each source's across-target mean per gene
(its generic response). Targets are restricted to those that respond clearly in CD4 (>= 30 responsive genes)."""
import sys
import numpy as np, pandas as pd
from vcc.effects import effect_matrix
from vcc.responsive import n_responsive_genes
from vcc.stats import load

vcc = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").target_gene.astype(str).tolist()
S = ["K562", "HCT116", "HEK293T", "CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr"]
M = {s: effect_matrix(s) for s in S}
resp = n_responsive_genes(load("CD4_Stim8hr")); strong = set(resp.index[resp >= 30]) & set(vcc)
print(f"VCC targets: {len(vcc)}; responsive in CD4 (>=30 genes): {len(strong)}")
expr = np.ones(len(M["K562"][2]), bool)
for s in S: expr &= (2 ** M[s][2] - 1) >= 5

def vec(s, targets, center):
    idx, E, _, meas = M[s]; ii = idx.get_indexer(targets); E = E[np.maximum(ii, 0)].copy(); E[ii < 0] = np.nan
    if center:
        full = np.nanmean(M[s][1][idx.get_indexer([t for t in vcc if t in idx])], 0, keepdims=True); E = E - full
    return E

def agree(a, b, center):
    both = [t for t in sorted(strong) if t in M[a][0] and t in M[b][0]]
    A, B = vec(a, both, center), vec(b, both, center)
    ok = expr & ~np.isnan(A).any(0) & ~np.isnan(B).any(0)
    r = [np.corrcoef(x[ok], y[ok])[0, 1] for x, y in zip(A, B) if x[ok].std() > 0 and y[ok].std() > 0]
    return np.mean(r) if r else np.nan, len(both)

for center, name in ((False, "RAW effects"), (True, "generic response removed (target-specific part)")):
    tab = pd.DataFrame({b: {a: agree(a, b, center)[0] for a in S} for b in S}).round(3)
    print(f"\n{name}: mean per-target Pearson between sources"); print(tab.to_string())
print("\ntargets used per pair (example K562-CD4_Stim8hr):", agree("K562", "CD4_Stim8hr", False)[1])
