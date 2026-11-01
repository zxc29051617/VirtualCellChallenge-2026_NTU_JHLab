"""Is a gene's tendency to respond to perturbations (excess variance of its noise-normalised deltas across
perturbations) shared between contexts? Prints the correlation matrix of per-gene responsiveness."""
import numpy as np, warnings
from vcc.contexts import load_all
warnings.filterwarnings("ignore")
C = load_all(["K562", "HCT116", "HEK293T", "H1", "CD4_Rest", "CD4_Stim8hr", "K562ess", "HepG2", "Jurkat"])
R = {}
for n, c in C.items():
    D = c.delta - np.nanmean(c.delta, 0, keepdims=True)
    z2 = D ** 2 / np.maximum(c.var, 1e-6)
    R[n] = np.log1p(np.maximum(np.nanmean(z2, 0) - 1, 0))       # excess over pure noise
    R[n][~c.measured | (c.ctrl < 0.5)] = np.nan
names = list(R)
print(" " * 13 + " ".join(f"{n[:7]:>7s}" for n in names))
for a in names:
    row = []
    for b in names:
        m = np.isfinite(R[a]) & np.isfinite(R[b]); row.append(np.corrcoef(R[a][m], R[b][m])[0, 1])
    print(f"{a:12s} " + " ".join(f"{x:7.2f}" for x in row))
