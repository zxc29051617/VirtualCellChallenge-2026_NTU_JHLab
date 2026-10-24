"""VCC 2025 H1 data -> stage-3 line file data/stage3/h1_H1.h5ad (raw counts, VCC 2026 gene axis).
Every perturbation of the three 2025 splits (at most 400 cells each) plus 20,000 controls, binomially
thinned so the control median is 20,000 UMI -- the depth the 2026 reference is downsampled to.
"""
import time
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, h5py
from anndata.io import sparse_dataset, write_elem

ROOT = "/home/zxc29051617/VCC/data"
genes = pd.read_csv(f"{ROOT}/vcc2026/gene_names.csv").gene_name.astype(str).tolist()
gidx = {g: i for i, g in enumerate(genes)}
vcc = set(pd.read_csv(f"{ROOT}/vcc2026/pert_counts.csv").target_gene.astype(str))
rng = np.random.default_rng(0)
files = {s: f"{ROOT}/external/h1_2025/adata_{s}.h5ad" for s in ["Training", "Validation", "Test"]}

# controls are spread over the three files: pick 20,000 in total, proportionally
ad_ = {s: ad.read_h5ad(f, backed="r") for s, f in files.items()}
nctrl = {s: int((a.obs.target_gene == "non-targeting").sum()) for s, a in ad_.items()}
quota = {s: int(round(20000 * n / sum(nctrl.values()))) for s, n in nctrl.items()}

# global thinning factor from a control sample
a = ad_["Validation"]; ci = np.flatnonzero((a.obs.target_gene == "non-targeting").values)[:3000]
med = np.median(np.asarray(sp.csr_matrix(a.X[np.sort(ci)]).sum(1)).ravel())
P = min(1.0, 20000 / med)
print(f"control median {med:.0f} UMI -> thinning p = {P:.3f}; control quota {quota}", flush=True)

out = f"{ROOT}/stage3/h1_H1.h5ad"
f = h5py.File(out + ".part", "w"); X = None; obs = []; nnz = 0; t0 = time.time()
measured = np.zeros(len(genes), bool)
for s, a in ad_.items():
    col = np.array([gidx.get(g, -1) for g in a.var_names.astype(str)]); ok = col >= 0
    measured[col[ok]] = True
    tg = a.obs.target_gene.astype(str).values
    keep = []
    for t in np.unique(tg):
        idx = np.flatnonzero(tg == t)
        n = quota[s] if t == "non-targeting" else 400
        keep.append(rng.permutation(idx)[:n])
    keep = np.sort(np.concatenate(keep))
    for lo in range(0, len(keep), 8000):        # ~8k nnz/cell: keep each chunk ~2 GB
        k = keep[lo:lo + 8000]
        m = sp.csr_matrix(a.X[k])[:, np.flatnonzero(ok)].tocoo()
        d = rng.binomial(m.data.astype(np.int64), P).astype(np.float32)
        m = sp.csr_matrix((d, (m.row, col[ok][m.col])), shape=(len(k), len(genes)))
        m.sum_duplicates(); m.eliminate_zeros()
        if X is None:
            write_elem(f, "X", m); X = sparse_dataset(f["X"])
        else:
            X.append(m)
        nnz += m.nnz
    o = a.obs.iloc[keep]
    t = o.target_gene.astype(str).values
    obs.append(pd.DataFrame({"cell_line": "H1", "target_gene": np.where(t == "non-targeting", "Non-Targeting", t),
                             "role": np.where(t == "non-targeting", "control",
                                              np.where(np.isin(t, list(vcc)), "vcc_target", "h1_target")),
                             "batch": o.batch.astype(str).values, "split": s}, index=o.index.astype(str)))
    print(f"  {s}: kept {len(keep):,} cells, nnz so far {nnz:,}, {time.time()-t0:.0f}s", flush=True)
obs = pd.concat(obs)
for c in ("cell_line", "target_gene", "role", "batch", "split"): obs[c] = obs[c].astype("category")
obs["is_control"] = obs.role == "control"
f.attrs["encoding-type"] = "anndata"; f.attrs["encoding-version"] = "0.1.0"
write_elem(f, "obs", obs); write_elem(f, "var", pd.DataFrame({"measured": measured}, index=pd.Index(genes)))
for key in ("obsm", "varm", "obsp", "varp", "layers"): write_elem(f, key, {})
write_elem(f, "uns", {"source": "Arc VCC 2025 H1 (CC0)", "thinning_p": P})
f.close()
import os; os.replace(out + ".part", out)
print(f"wrote {out}: {len(obs):,} cells ({obs.role.value_counts().to_dict()}), nnz {nnz:,}, "
      f"{measured.sum()} VCC genes measured")
