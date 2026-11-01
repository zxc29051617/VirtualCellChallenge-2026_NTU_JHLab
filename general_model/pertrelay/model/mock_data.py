"""Mock exam data from H1 (Arc VCC 2025: 400 cells per target, like D/E/F) in the challenge layout (2026-10-22).

    .venv/bin/python -I model/mock_data.py        (from pertrelay/)

data/mock_h1/context_H.h5ad   4,000 H1 control cells (seed 0), obs target_gene = non-targeting, context = H
data/mock_h1/s<k>/            k = 0..4: ~60 targets each (H1 targets with >= 50 cells that another cell type perturbed;
                              shuffled, seed 0): pert_counts.csv, gene_names.csv, manifest.json, context_H.h5ad (link),
                              real.h5ad = up to 400 cells per target + the same 4,000 controls (the scorer's reference)
Five subsets keep the official scorer within this machine's 15 GB; H1 is excluded from every source (VCC_EXCLUDE=H1).
"""
import json, os, shutil, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

ROOT = "/home/zxc29051617/VCC"; OUT = f"{ROOT}/data/mock_h1"; os.makedirs(OUT, exist_ok=True)
A = ad.read_h5ad(f"{ROOT}/data/stage3/h1_H1.h5ad", backed="r")
genes = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv")["gene_name"].astype(str).tolist()
assert [str(g) for g in A.var_names] == genes, "gene axis differs"
lab = A.obs["target_gene"].astype(str).values; ctrl = A.obs["is_control"].values.astype(bool)
rng = np.random.default_rng(0)
ci = np.sort(rng.choice(np.flatnonzero(ctrl), 4000, replace=False))
Xc = sp.csr_matrix(A.X[ci]).astype(np.float32)
C = ad.AnnData(Xc, obs=pd.DataFrame({"target_gene": "non-targeting", "context": "H"}, index=[f"c{i}" for i in range(len(ci))]),
               var=pd.DataFrame(index=genes))
C.write_h5ad(f"{OUT}/context_H.h5ad")

train = [c for c in B.CONTEXTS if c not in B.FOLDS["H1"]]
targets = B.eval_targets("H1", train)                     # same rule as the benchmark: >= 50 cells, perturbed elsewhere
targets = list(rng.permutation(targets)); parts = np.array_split(np.array(targets), 5)
for k, part in enumerate(parts):
    d = f"{OUT}/s{k}"; os.makedirs(d, exist_ok=True)
    pd.DataFrame({"target_gene": part}).to_csv(f"{d}/pert_counts.csv", index=False)
    shutil.copy(f"{ROOT}/data/vcc2026/gene_names.csv", f"{d}/gene_names.csv")
    json.dump({"season": "mock", "partition": "mock", "panel_id": f"mock-h1-s{k}", "contexts": ["H"], "pert_col": "target_gene",
               "context_col": "context", "control_label": "non-targeting", "n_genes": len(genes), "cells_per_pert": 400}, open(f"{d}/manifest.json", "w"))
    if os.path.lexists(f"{d}/context_H.h5ad"): os.remove(f"{d}/context_H.h5ad")
    os.symlink(f"{OUT}/context_H.h5ad", f"{d}/context_H.h5ad")
    rows = []
    for t in part:
        r = np.flatnonzero(lab == t)
        rows.append(np.sort(rng.choice(r, 400, replace=False)) if len(r) > 400 else r)
    rows = np.concatenate(rows); order = np.argsort(rows); rows_sorted = rows[order]
    Xp = sp.csr_matrix(A.X[rows_sorted]).astype(np.float32)
    R = ad.AnnData(sp.vstack([Xc, Xp]).tocsr(), obs=pd.DataFrame({"target_gene": np.concatenate([np.full(len(ci), "non-targeting"), lab[rows_sorted]])},
                   index=[f"r{i}" for i in range(len(ci) + len(rows_sorted))]), var=pd.DataFrame(index=genes))
    R.write_h5ad(f"{d}/real.h5ad")
    n = pd.Series(lab[rows_sorted]).value_counts()
    print(f"s{k}: {len(part)} targets, truth cells per target median {int(n.median())} (min {n.min()}), {R.n_obs:,} rows", flush=True)
