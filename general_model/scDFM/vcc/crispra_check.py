"""Do CRISPRa (activation) effects on VCC targets relate to CRISPRi (knockdown) effects, and with which sign?
Hs27 / RPE1 mean-population matrices: rows = guides, X = effect on each measured gene (checked below)."""
import anndata as ad, numpy as np, pandas as pd
from vcc.effects import effect_matrix
vcc = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").target_gene.astype(str).tolist()
a = ad.read_h5ad("/home/zxc29051617/VCC/data/external/crispra/Hs27_mean_pop.h5ad")
X = np.asarray(a.X); print("X range: min %.3f max %.3f mean %.4f; sd %.3f" % (X.min(), X.max(), X.mean(), X.std()))
print("var 'mean' range:", float(a.var['mean'].min()), float(a.var['mean'].max()), "| target_expr ex:", a.obs.target_expr.iloc[:3].tolist(), "| active:", a.obs.active.value_counts().to_dict())
genes = np.array(a.var_names)
tg = a.obs.target_gene.astype(str).values
for src in ["CD4_all", "K562"]:
    from vcc.stats import load
    idx, E, c, meas = effect_matrix(src) if src != "CD4_all" else effect_matrix("CD4_Stim8hr")
    gi = {g: i for i, g in enumerate(pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name)}
    cols = np.array([gi.get(g, -1) for g in genes]); ok = (cols >= 0)
    rs = []
    for t in sorted(set(tg) & set(vcc)):
        if t not in idx: continue
        rows = np.flatnonzero((tg == t) & a.obs.active.astype(bool).values) if "active" in a.obs else np.flatnonzero(tg == t)
        if not len(rows): rows = np.flatnonzero(tg == t)
        v = X[rows].mean(0)[ok]; w = E[idx.get_loc(t)][cols[ok]]
        keep = ~np.isnan(w) & (np.abs(v) > 0)
        if keep.sum() > 200 and v[keep].std() > 0 and w[keep].std() > 0: rs.append(np.corrcoef(v[keep], w[keep])[0, 1])
    rs = np.array(rs)
    print(f"{src}: {len(rs)} VCC targets compared; mean r {rs.mean():+.3f}, median {np.median(rs):+.3f}; negative {np.mean(rs<0):.0%}; |r|>0.1: {np.mean(np.abs(rs)>0.1):.0%}; r<-0.1: {np.mean(rs<-0.1):.0%}  r>+0.1: {np.mean(rs>0.1):.0%}")
