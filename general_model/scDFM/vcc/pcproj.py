"""Post-process an effects file: keep only the part of each predicted response that lies along the new context's own
axes of natural variation (top principal components of its control cells), optionally rescaling it.

Rationale: perturbation responses mostly move cells along programs a cell type already varies along (cell cycle,
stress, ...). Projecting a transferred prediction onto the target context's control PCs removes transfer noise in
directions that context does not vary in. Own-gene and cis-neighbour effects are kept as they are.

    python -m vcc.pcproj <effects.npz> <controls.h5ad | context letter> <out.npz> --k 50 --scale 1.0 [--trans-scale 1]
"""
import argparse
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from vcc.genome import on_axis
from vcc.gm import CIS_BINS

p = argparse.ArgumentParser()
p.add_argument("effects"); p.add_argument("controls"); p.add_argument("out")
p.add_argument("--k", type=int, default=50, help="number of control PCs to keep (0 = no projection)")
p.add_argument("--scale", type=float, default=1.0, help="multiply the projected trans part")
p.add_argument("--min-mean", type=float, default=0.05, help="genes with mean CP10k below this are left out of the PCA")
p.add_argument("--max-cells", type=int, default=8000)
a = p.parse_args()

genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).values
src = a.controls if a.controls.endswith(".h5ad") else f"/home/zxc29051617/VCC/data/vcc2026/context_{a.controls}.h5ad"
A = ad.read_h5ad(src)
if "target_gene" in A.obs: A = A[A.obs.target_gene.astype(str).isin(["non-targeting", "Non-Targeting"])]
X = sp.csr_matrix(A.X)
rng = np.random.default_rng(0)
if X.shape[0] > a.max_cells: X = X[np.sort(rng.choice(X.shape[0], a.max_cells, replace=False))]
lib = np.asarray(X.sum(1)).ravel()
N = sp.diags(1e4 / np.maximum(lib, 1)) @ X; N.data = np.log1p(N.data)
mean_cp = np.asarray((sp.diags(1e4 / np.maximum(lib, 1)) @ X).mean(0)).ravel()   # CP10k per gene
use = mean_cp >= a.min_mean
L = N[:, use].toarray(); L -= L.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(L, full_matrices=False)
V = Vt[:a.k].T if a.k > 0 else None                                       # genes(use) x k
print(f"controls {X.shape[0]} cells, {use.sum()} genes in PCA, top-{a.k} PCs explain "
      f"{(S[:a.k] ** 2).sum() / (S ** 2).sum():.1%} of control variance", flush=True)

z = np.load(a.effects, allow_pickle=True); targets = z["targets"]; lfc = z["lfc"].astype(np.float64)
gi = {g: i for i, g in enumerate(genes)}
chr_, tss, strand = on_axis(genes)
x = mean_cp                                                                # control CP10k
base = np.log1p(x)
out = lfc.copy()
for t_i, t in enumerate(targets):
    fc = 2.0 ** np.nan_to_num(lfc[t_i])
    keep = np.zeros(len(genes), bool)                                     # own + cis genes are not projected
    k = gi.get(str(t))
    if k is not None:
        keep[k] = True
        if tss[k] >= 0:
            keep |= (chr_ == chr_[k]) & (tss >= 0) & (np.abs(tss - tss[k]) < CIS_BINS[-1])
    delta = np.log1p(x * fc) - base                                       # predicted shift in log1p(CP10k)
    d = delta.copy(); d[keep] = 0
    if V is not None:
        du = d[use]; d_proj = np.zeros_like(d); d_proj[use] = V @ (V.T @ du)
    else:
        d_proj = d
    d_new = a.scale * d_proj; d_new[keep] = delta[keep]
    fc_new = np.expm1(base + d_new) / np.maximum(x, 1e-9)
    fc_new[x <= 0] = 1.0
    out[t_i] = np.log2(np.clip(fc_new, 1e-3, 1e3))
np.savez_compressed(a.out, targets=targets, lfc=out.astype(np.float32))
print(f"wrote {a.out}: {len(targets)} targets (k={a.k}, scale={a.scale})")
