"""Which DepMap cell lines do the challenge contexts look like? (2026-10-25)

    python -I depmap_match.py <OmicsExpressionProteinCodingGenesTPMLogp1.csv> <Model.csv>

Control pseudobulks of A-F (log2 CPM + 1) against CCLE RNA-seq (log2 TPM + 1) on the 3,000 genes that vary most across
cell lines: Pearson correlation of profiles centred on the CCLE mean. Prints the 12 best-matching lines per context.
"""
import sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

expr_path, model_path = sys.argv[1], sys.argv[2]
E = pd.read_csv(expr_path, index_col=0)
if "ModelID" in E.columns: E = E.set_index("ModelID")
E = E.loc[:, [c for c in E.columns if "(" in c]]
E.columns = [c.split(" (")[0] for c in E.columns]
E = E.loc[:, ~E.columns.duplicated()]
M = pd.read_csv(model_path, index_col=0)
genes = [l.strip().split(",")[0] for l in open("/home/zxc29051617/VCC/data/vcc2026_final/gene_names.csv")][1:]
common = [g for g in genes if g in E.columns]
gi = {g: i for i, g in enumerate(genes)}
P = {}
for d, cs in (("vcc2026", "ABC"), ("vcc2026_final", "DEF")):
    for c in cs:
        X = sp.csr_matrix(ad.read_h5ad(f"/home/zxc29051617/VCC/data/{d}/context_{c}.h5ad").X)
        s = np.asarray(X.sum(0)).ravel(); P[c] = np.log2(1e6 * s / s.sum() + 1)
C = E[common].to_numpy(np.float64)
top = np.argsort(-C.var(0))[:3000]
mu = C[:, top].mean(0); Cc = C[:, top] - mu
Cc /= np.linalg.norm(Cc, axis=1, keepdims=True)
names = M.reindex(E.index)
for c, v in P.items():
    x = v[[gi[common[j]] for j in top]] - mu; x /= np.linalg.norm(x)
    r = Cc @ x; best = np.argsort(-r)[:12]
    print(f"\n{c}: " + "; ".join(f"{names.iloc[i].get('StrippedCellLineName', E.index[i])} "
                                 f"({names.iloc[i].get('OncotreePrimaryDisease', '?')}) {r[i]:.2f}" for i in best))
