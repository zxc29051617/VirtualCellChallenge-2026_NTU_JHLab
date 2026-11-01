"""What do contexts A/B/C look like? Control-profile correlation with every training context + marker genes."""
import json, sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
ROOT = "/home/zxc29051617/VCC"; sys.path.insert(0, f"{ROOT}/pertrelay/experiments/signature_20261013"); import sigexp as S
C = S.load_cache()
genes = S.GENES; GI = S.GI
prof = {c: (C[c]["ctrl"].astype(float), C[c]["measured"].astype(bool)) for c in S.CONTEXTS}
for x in "ABC":
    A = ad.read_h5ad(f"{ROOT}/data/vcc2026/context_{x}.h5ad"); s = np.asarray(sp.csr_matrix(A.X).sum(0)).ravel().astype(float)
    prof[x] = (np.log1p(5e4 * s / s.sum()), s > 0)
mu = np.mean([prof[c][0] for c in S.CONTEXTS], 0)
def r(a, b):
    la, ma = prof[a]; lb, mb = prof[b]; m = ma & mb & ((np.expm1(la) >= 1) | (np.expm1(lb) >= 1))
    x, y = la[m] - mu[m], lb[m] - mu[m]; return float(np.corrcoef(x, y)[0, 1])
rows = {x: {c: round(r(x, c), 3) for c in S.CONTEXTS} for x in "ABC"}
print(pd.DataFrame(rows).to_string())
print("A-B", round(r("A", "B"), 3), "A-C", round(r("A", "C"), 3), "B-C", round(r("B", "C"), 3))
mk = ["POU5F1", "NANOG", "SOX2", "LIN28A", "CD3E", "CD4", "PTPRC", "HBG1", "GATA1", "ALB", "AFP", "KRT18", "KRT8", "VIM", "PAX6", "NES",
      "SOX17", "FOXA2", "GATA4", "GATA6", "HAND1", "TBXT", "MIXL1", "COL1A1", "PECAM1", "CDH5", "NKX2-5", "TNNT2", "SOX10", "MKI67", "EOMES", "T", "CDX2", "HNF1B", "PDX1"]
cols = ["A", "B", "C", "H1", "KOLF", "Jurkat", "K562", "HepG2", "HCT116"]
tab = pd.DataFrame({c: [round(float(np.expm1(prof[c][0][GI[g]])), 1) if g in GI and prof[c][1][GI[g]] else np.nan for g in mk] for c in cols}, index=mk)
print(tab.to_string())
