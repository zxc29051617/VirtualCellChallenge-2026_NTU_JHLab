"""Extract the stage-3 cells from a Replogle 2022 raw single-cell h5ad, on the VCC gene axis.
Same selection as tools/select_cells.py uses for X-Atlas: every cell of a VCC target, at most
64 cells of each auxiliary gene, 20,000 random controls. Genes Replogle did not keep are flagged
var.measured = False (stored as zero but must not be read as zero expression).

    python extract_replogle.py K562_gwps_raw_singlecell_01.h5ad K562 replogle_K562_gwps
"""
import sys, time
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, h5py
from anndata.io import sparse_dataset, write_elem

src, LINE, name = sys.argv[1:4]
ROOT = "/home/zxc29051617/VCC/data"
genes = pd.read_csv(f"{ROOT}/vcc2026/gene_names.csv").gene_name.astype(str).tolist()
gidx = {g: i for i, g in enumerate(genes)}
vcc = set(pd.read_csv(f"{ROOT}/vcc2026/pert_counts.csv").target_gene.astype(str))
aux = set(open(f"{ROOT}/coverage/aux_genes.txt").read().split())
h1 = set(pd.concat([pd.read_csv(f"{ROOT}/external/meta/h1_pert_counts_{s}.csv") for s in ["Training", "Validation", "Test"]])
         .target_gene.astype(str)) - vcc

A = ad.read_h5ad(f"{ROOT}/external/replogle2022/{src}", backed="r")
col = np.array([gidx.get(g, -1) for g in A.var.gene_name.astype(str)])
measured = np.zeros(len(genes), bool); measured[col[col >= 0]] = True
g = A.obs.gene.astype(str).values
rng = np.random.default_rng(0)
order = rng.permutation(A.n_obs)
role = np.full(A.n_obs, "", dtype=object)
role[np.isin(g, list(vcc))] = "vcc_target"
role[np.isin(g, list(h1))] = "h1_target"
seen = {}
for i in order:                                   # first 64 cells (random order) per aux gene
    if g[i] in aux and seen.get(g[i], 0) < 64:
        role[i] = "aux"; seen[g[i]] = seen.get(g[i], 0) + 1
ctrl = order[g[order] == "non-targeting"][:20000]; role[ctrl] = "control"
keep = np.flatnonzero(role != "")
print(f"{LINE}: keep {len(keep):,} of {A.n_obs:,} cells: "
      f"{pd.Series(role[keep]).value_counts().to_dict()}; {measured.sum()} VCC genes measured", flush=True)

out = f"{ROOT}/stage3/{name}.h5ad"
f = h5py.File(out + ".part", "w"); X = None; nnz = 0; t0 = time.time()
okc = col >= 0
for lo in range(0, A.n_obs, 50000):
    hi = min(A.n_obs, lo + 50000); k = keep[(keep >= lo) & (keep < hi)] - lo
    if len(k):
        m = sp.csr_matrix(A.X[lo:hi])[k][:, np.flatnonzero(okc)].tocoo()
        m = sp.csr_matrix((m.data, (m.row, col[okc][m.col])), shape=(len(k), len(genes)), dtype=np.float32)
        m.sum_duplicates(); m.eliminate_zeros()
        if X is None:
            write_elem(f, "X", m); X = sparse_dataset(f["X"])
        else:
            X.append(m)
        nnz += m.nnz
    print(f"  rows {hi:,}/{A.n_obs:,}  nnz {nnz:,}  {time.time()-t0:.0f}s", flush=True)
o = A.obs.iloc[keep]
obs = pd.DataFrame({"cell_line": LINE, "target_gene": np.where(role[keep] == "control", "Non-Targeting", g[keep]),
                    "role": role[keep], "batch": o.gem_group.astype(str).values,
                    "total_counts": o.UMI_count.values}, index=o.index.astype(str))
for c in ("cell_line", "target_gene", "role", "batch"): obs[c] = obs[c].astype("category")
obs["is_control"] = obs.role == "control"
f.attrs["encoding-type"] = "anndata"; f.attrs["encoding-version"] = "0.1.0"
write_elem(f, "obs", obs); write_elem(f, "var", pd.DataFrame({"measured": measured}, index=pd.Index(genes)))
for key in ("obsm", "varm", "obsp", "varp", "layers"): write_elem(f, key, {})
write_elem(f, "uns", {"source": f"Replogle et al. 2022 {src} (CC BY 4.0)"})
f.close()
import os; os.replace(out + ".part", out)
print(f"wrote {out}: {len(obs):,} cells, nnz {nnz:,}")
