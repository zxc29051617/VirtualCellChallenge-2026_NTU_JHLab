import numpy as np, pandas as pd
from vcc.stats import load, lfc_table
vcc = set(pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").target_gene)
S = {l: load(l) for l in ["HCT116", "HEK293T"]}
T = {l: lfc_table(S[l]) for l in S}
genes = S["HCT116"]["genes"]
# expressed genes: >= 5 CPM in controls of both lines
expr = (T["HCT116"][2] >= 5) & (T["HEK293T"][2] >= 5)
print(f"genes >=5 CPM in both lines' controls: {expr.sum()}")
for l in S:
    names, L, c = T[l]
    isv = np.array([n in vcc for n in names])
    strong = (np.abs(np.nan_to_num(L[:, expr])) > 0.5).sum(1)
    print(f"{l}: genes |lfc|>0.5 among expressed — VCC targets median {np.median(strong[isv]):.0f} "
          f"(>=20 genes: {np.mean(strong[isv]>=20):.0%}); aux median {np.median(strong[~isv]):.0f}")
# cross-line agreement for shared perturbations
a_n, a_L, _ = T["HCT116"]; b_n, b_L, _ = T["HEK293T"]
shared = sorted(set(a_n) & set(b_n)); ia = [a_n.index(x) for x in shared]; ib = [b_n.index(x) for x in shared]
A = np.nan_to_num(a_L[ia][:, expr]); B = np.nan_to_num(b_L[ib][:, expr])
A -= A.mean(0); B -= B.mean(0)   # remove each line's shared/generic response
An = A / np.linalg.norm(A, axis=1, keepdims=True); Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
C = An @ Bn.T
diag = np.diag(C); off = C[~np.eye(len(shared), dtype=bool)]
rank = (C > diag[:, None]).sum(1) / (len(shared) - 1)
isv = np.array([s in vcc for s in shared])
print(f"shared perturbations {len(shared)} (VCC targets {isv.sum()}): matched cos {diag.mean():.3f} vs mismatched {off.mean():.3f}")
print(f"  PDS-like (1 - rank of own match): all {1-rank.mean():.3f}; VCC targets {1-rank[isv].mean():.3f}; aux {1-rank[~isv].mean():.3f}")
# how concentrated is the response: |lfc| mass captured by top-K genes chosen from HCT116, measured on HEK293T
w_a = np.abs(A).mean(0); w_b = np.abs(B).mean(0); order = np.argsort(-w_a)
for K in (500, 1000, 2000, 3000, 5000):
    print(f"  top-{K} genes by HCT116 response cover {w_b[order[:K]].sum()/w_b.sum():.0%} of HEK293T response mass")
