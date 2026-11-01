"""A single-cell perturbation h5ad (raw counts, e.g. scPerturb) -> a model_v2 context (2026-10-23).

    .venv/bin/python -I model/sc_build.py <h5ad> <name> [--pert-col perturbation] [--control control]

Cells with exactly one perturbation (obs nperts == 1 when the column exists) are summed per target on the VCC gene axis
(var names = symbols; duplicates summed); controls = the control label (nperts == 0). Targets with >= 20 cells. Then the
same transform as build_data.py (vcc.contexts.build + vcc.gm._ratios: CP50k ln ratios, NaN below X_MIN or unmeasured,
minus the context's mean response, Poisson-delta variance). Writes data/stage3/model_v2/<name>.npz.
"""
import argparse, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
from vcc.contexts import build
from vcc.gm import _ratios

p = argparse.ArgumentParser(); p.add_argument("h5ad"); p.add_argument("name")
p.add_argument("--pert-col", default="perturbation"); p.add_argument("--control", default="control")
a = p.parse_args()
genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); gi = {g: i for i, g in enumerate(genes)}
A = ad.read_h5ad(a.h5ad, backed="r")
lab = A.obs[a.pert_col].astype(str).values
single = A.obs["nperts"].astype(int).values == 1 if "nperts" in A.obs else np.ones(len(lab), bool)
is_ctrl = lab == a.control
keep = single | is_ctrl
names = sorted(set(lab[single & ~is_ctrl])); ti = {t: k + 1 for k, t in enumerate(names)}       # row 0 = __ctrl__
grp = np.array([0 if c else ti.get(t, -1) for t, c in zip(lab, is_ctrl)]); grp[~keep] = -1
col = np.array([gi.get(str(g), -1) for g in A.var_names]); ok = col >= 0
measured = np.zeros(len(genes), bool); measured[col[ok]] = True
sums = np.zeros((len(names) + 1, len(genes))); ncell = np.bincount(grp[grp >= 0], minlength=len(names) + 1)
for lo in range(0, A.n_obs, 20000):
    hi = min(A.n_obs, lo + 20000); g = grp[lo:hi]; m = g >= 0
    if not m.any(): continue
    X = sp.csr_matrix(A.X[lo:hi])[np.flatnonzero(m)][:, np.flatnonzero(ok)]
    ind = sp.csr_matrix((np.ones(m.sum()), (g[m], np.arange(m.sum()))), shape=(len(names) + 1, m.sum()))
    S = (ind @ X).toarray(); np.add.at(sums.T, col[ok], S.T)
st = {"names": np.array(["__ctrl__"] + names), "sums": sums.astype(np.float32), "ncell": ncell, "measured": measured, "genes": np.array(genes)}
c = build(a.name, st, min_cells=20, keep_var=True)
L, xc = _ratios(c)
cm = np.nanmean(L, 0).astype(np.float32)
own = np.array([L[j, gi[t]] if t in gi else np.nan for j, t in enumerate(c.perts)], np.float32)
out = f"/home/zxc29051617/VCC/data/stage3/model_v2/{a.name}.npz"
np.savez(out, perts=c.perts.astype(str), ncell=c.ncell, Y=(L - cm[None]).astype(np.float16), var=c.var.astype(np.float16), own_obs=own,
         ctrl=c.ctrl.astype(np.float32), measured=c.measured, cm=cm, genes=np.array(genes))
print(f"{a.name}: {len(c.perts)} targets (>= 20 single-perturbation cells), {ncell[0]} control cells, cells per target median "
      f"{int(np.median(c.ncell))}, knockdown median {np.nanmedian(own):+.2f}, usable {np.isfinite(L).mean():.2f}, measured genes {measured.sum()}")
