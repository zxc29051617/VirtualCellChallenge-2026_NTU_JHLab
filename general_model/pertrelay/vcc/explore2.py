import numpy as np
from vcc.stats import load, signal_variance, lfc_table
S = {l: load(l) for l in ["HCT116", "HEK293T"]}
sv = {l: signal_variance(S[l]) for l in S}
c = {l: lfc_table(S[l])[2] for l in S}
expr = (c["HCT116"] >= 5) & (c["HEK293T"] >= 5)
for l in S:
    v = np.where(expr, sv[l], np.nan)
    print(f"{l}: genes with positive signal variance {np.sum(v > 0)} / {expr.sum()} expressed")
a = np.nan_to_num(np.where(expr, sv["HCT116"], -np.inf), neginf=-np.inf)
b = np.clip(np.nan_to_num(np.where(expr, sv["HEK293T"], 0)), 0, None)
order = np.argsort(-a)
print(f"rank corr of signal variance across lines (expressed genes): "
      f"{np.corrcoef(np.argsort(np.argsort(sv['HCT116'][expr])), np.argsort(np.argsort(sv['HEK293T'][expr])))[0,1]:.3f}")
for K in (500, 1000, 2000, 3000, 5000):
    print(f"top-{K} by HCT116 signal variance capture {b[order[:K]].sum()/b.sum():.0%} of HEK293T positive signal variance")
