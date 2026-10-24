"""RPE1 (Replogle 2022 essential-gene CRISPRi Perturb-seq, the kept bulk summary) as a model_v2 context (2026-10-23).

    .venv/bin/python -I repro/model_v2_20261022/rpe1_build.py        (from scDFM/)

The single-cell file was deleted in the 10-21 clean-up; the bulk file holds, per guide group, the mean raw UMI per gene
over the filtered cells and their number. Sums = mean x cells (exact for K562, checked 10-22). Controls = the
non-targeting groups (mean x unfiltered cells, since their filtered count is missing). Then the same transform as
build_data.py / vcc.gm._ratios: CP50k, ln((x_p + 0.5) / (x_c + 0.5)), NaN below X_MIN in the controls or unmeasured,
minus the context's mean response; variance from the Poisson delta method as vcc.contexts.build.
"""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

OUT = "/home/zxc29051617/VCC/data/stage3/model_v2/RPE1.npz"; EPS, X_MIN, BULK = 0.5, 0.25, 5e4
genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); gi = {g: i for i, g in enumerate(genes)}
A = ad.read_h5ad("/home/zxc29051617/VCC/data/external/replogle2022/rpe1_raw_bulk_01.h5ad")
X = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X, np.float64)
col = np.array([gi.get(g, -1) for g in A.var["gene_name"].astype(str)]); ok = col >= 0
measured = np.zeros(len(genes), bool); measured[col[ok]] = True
tg = np.array([s.split("_")[1] for s in A.obs_names]); nf = A.obs["num_cells_filtered"].values; nu = A.obs["num_cells_unfiltered"].values
ctrl_rows = np.flatnonzero(tg == "non-targeting")
c_sum = np.zeros(len(genes)); c_sum[col[ok]] = (X[ctrl_rows][:, ok] * nu[ctrl_rows, None]).sum(0); n_ctrl = int(nu[ctrl_rows].sum())
perts, sums, ncell = [], [], []
for t in sorted(set(tg) - {"non-targeting"}):
    r = np.flatnonzero((tg == t) & np.isfinite(nf))
    if not len(r) or nf[r].sum() < 20: continue
    s = np.zeros(len(genes)); s[col[ok]] = (X[r][:, ok] * nf[r, None]).sum(0)
    perts.append(t); sums.append(s); ncell.append(int(round(nf[r].sum())))
S = np.array(sums)
lognorm = lambda s: np.log1p(BULK * s / np.maximum(s.sum(-1, keepdims=True), 1))
ctrl = lognorm(c_sum[None])[0]; delta = lognorm(S) - ctrl[None]
xc = np.expm1(ctrl); xp = np.expm1(ctrl[None] + delta)
L = np.log((xp + EPS) / (xc + EPS)); L[:, ~(measured & (xc >= X_MIN))] = np.nan
cm = np.nanmean(L, 0)
v = lambda s: (BULK / np.maximum(s.sum(-1, keepdims=True), 1)) ** 2 * s / (1 + BULK * s / np.maximum(s.sum(-1, keepdims=True), 1)) ** 2
var = v(S) + v(c_sum[None])
own = np.array([L[i, gi[t]] if t in gi else np.nan for i, t in enumerate(perts)], np.float32)
np.savez(OUT, perts=np.array(perts), ncell=np.array(ncell), Y=(L - cm[None]).astype(np.float16), var=var.astype(np.float16), own_obs=own,
         ctrl=ctrl.astype(np.float32), measured=measured, cm=np.nan_to_num(cm).astype(np.float32), genes=np.array(genes))
print(f"RPE1: {len(perts)} targets ({n_ctrl} control cells), cells per target median {int(np.median(ncell))}, knockdown median {np.nanmedian(own):+.2f}, usable {np.isfinite(L).mean():.2f}")
