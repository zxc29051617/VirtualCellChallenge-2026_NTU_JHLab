"""Which public cell line does each context's control profile resemble? (Human Protein Atlas cell-line RNA, nTPM)

Validation first on training contexts whose identity is known, then A/B/C. Score: Pearson correlation of
(log1p context profile - HPA cross-line mean) with (log1p line profile - HPA cross-line mean) over the 3,000 genes that
vary most across HPA lines and are measured in the context.
"""
import sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
ROOT = "/home/zxc29051617/VCC"; sys.path.insert(0, f"{ROOT}/scDFM/repro/signature_20261013"); import sigexp as S
H = pd.read_csv(f"{ROOT}/data/external/hpa/rna_celline.tsv", sep="\t", usecols=["Gene name", "Cell line", "nTPM"],
                dtype={"Gene name": "category", "Cell line": "category", "nTPM": "float32"})
M = H.pivot_table(index="Gene name", columns="Cell line", values="nTPM", aggfunc="mean", observed=True)
L = np.log1p(M.astype(float)); print("HPA:", L.shape[1], "cell lines,", L.shape[0], "genes", flush=True)
C = S.load_cache(); prof = {c: (C[c]["ctrl"].astype(float), C[c]["measured"].astype(bool)) for c in S.CONTEXTS}
for x in "ABC":
    A = ad.read_h5ad(f"{ROOT}/data/vcc2026/context_{x}.h5ad"); s = np.asarray(sp.csr_matrix(A.X).sum(0)).ravel().astype(float)
    prof[x] = (np.log1p(1e6 * s / s.sum()), s > 0)
common = [g for g in S.GENES if g in L.index]; Lc = L.loc[common]; mean = Lc.mean(1).values; var = Lc.var(1).values
gidx = np.array([S.GI[g] for g in common])
for ctx in ["K562", "VIPK562", "HepG2", "HCT116", "HEK293T", "Jurkat", "H1", "KOLF", "A", "B", "C"]:
    l, m = prof[ctx]; l = np.log1p(np.expm1(l) * (1e6 / np.expm1(l).sum()))       # to log1p CPM like nTPM scale
    ok = m[gidx]; top = np.argsort(-np.where(ok, var, -1))[:3000]
    q = l[gidx][top] - mean[top]; R = Lc.values[top] - mean[top][:, None]
    r = (q - q.mean()) @ (R - R.mean(0)) / (np.linalg.norm(q - q.mean()) * np.linalg.norm(R - R.mean(0), axis=0))
    o = np.argsort(-r)[:5]
    print(f"{ctx:8s}", ", ".join(f"{Lc.columns[i]} {r[i]:.2f}" for i in o), flush=True)
