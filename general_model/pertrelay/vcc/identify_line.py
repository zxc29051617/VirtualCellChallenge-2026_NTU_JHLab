"""Which known cell lines does a set of control cells resemble? Compares the mean expression profile of a context's
control cells (log1p CPM) with every DepMap cell line (bulk RNA-seq, log1p TPM) by Spearman correlation over genes
expressed in the context. Generic: works for any new cell line, e.g. the final-test contexts D/E/F.
Sanity check: lines whose identity we know (K562, HCT116, HEK293T) must come out on top."""
import sys
import anndata as ad, numpy as np, pandas as pd
from scipy.stats import rankdata

D = "/home/zxc29051617/VCC/data/external/depmap"
model = pd.read_csv(f"{D}/Model.csv").set_index("ModelID")
prof = pd.read_csv(f"{D}/DefaultProfiles.csv")
expr = pd.read_csv(f"{D}/Expression_ProteinCoding_TPMLogp1.csv", index_col=0, dtype={0: str})
expr = expr.astype(np.float32)
expr.columns = [c.split(" (")[0] for c in expr.columns]
expr = expr.loc[:, ~expr.columns.duplicated()]
# one row per model: use the default profile when the table is indexed by profile id
idmap = dict(zip(prof.iloc[:, 0], prof.iloc[:, 1])) if prof.shape[1] > 1 else {}
rows = {i: (i if i in model.index else idmap.get(i)) for i in expr.index}
keep = [i for i, m in rows.items() if m in model.index]
expr = expr.loc[keep]; expr.index = [rows[i] for i in keep]
expr = expr[~expr.index.duplicated()]
print(f"DepMap: {expr.shape[0]} cell lines x {expr.shape[1]} genes")

genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist()
def profile(sums):
    cpm = sums / sums.sum() * 1e6
    return pd.Series(np.log1p(cpm), index=genes)

def match(name, p, top=6):
    common = [g for g in expr.columns if g in p.index and p[g] > np.log1p(5)]        # expressed (>= 5 CPM) in the context
    x = rankdata(p[common].values)
    E = expr[common].values
    R = np.apply_along_axis(rankdata, 1, E)
    r = (R - R.mean(1, keepdims=True)) @ (x - x.mean()) / (np.linalg.norm(R - R.mean(1, keepdims=True), axis=1) * np.linalg.norm(x - x.mean()))
    order = np.argsort(-r)[:top]
    print(f"\n{name}  ({len(common)} genes)")
    for i in order:
        m = model.loc[expr.index[i]]
        print(f"   r={r[i]:.3f}  {m.CellLineName:18s} {str(m.OncotreeLineage)[:18]:18s} {str(m.OncotreePrimaryDisease)[:34]}")

sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
from vcc.stats import load
for l in ["K562", "HCT116", "HEK293T"]:
    st = load(l); names = list(st["names"]); match(f"[known] {l}", profile(st["sums"][names.index("__ctrl__")].astype(float)))
for c in "ABC":
    X = ad.read_h5ad(f"/home/zxc29051617/VCC/data/vcc2026/context_{c}.h5ad").X
    match(f"[practice] context {c}", profile(np.asarray(X.sum(0)).ravel().astype(float)))
