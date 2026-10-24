"""More mock exams (2026-10-23): HCT116 and KOLF200 hold-outs in the same layout as mock_data.py (H1), so every change can
be checked with the official scorer on three hidden cell types.

    .venv/bin/python -I repro/model_v2_20261022/mock_data_loco.py [HCT116,KOLF200]        (from scDFM/)

From data/stage3/loco/<X>/real.h5ad (4,000 controls + perturbed cells, built 10-11 for the hold-out experiments):
data/mock_<x>/context_H.h5ad   the 4,000 controls (obs target_gene = non-targeting, context = H)
data/mock_<x>/s<k>/            k = 0..3: targets with >= 50 cells that a training context outside the hold-out's group
                               perturbed (shuffled, seed 0), up to 400 cells each; pert_counts.csv, gene_names.csv,
                               manifest.json, context_H.h5ad (link), real.h5ad (controls + the subset's cells)
mock_exam.py with MOCK_SET=hct116 / kolf200 uses them (VCC_EXCLUDE and the model's training contexts set accordingly).
"""
import json, os, shutil, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

ROOT = "/home/zxc29051617/VCC"
SETS = {"HCT116": "XAtlas", "KOLF200": "iPSC"}
genes = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv")["gene_name"].astype(str).tolist()
for held in (sys.argv[1] if len(sys.argv) > 1 else "HCT116,KOLF200").split(","):
    OUT = f"{ROOT}/data/mock_{held.lower()}"; os.makedirs(OUT, exist_ok=True)
    R = ad.read_h5ad(f"{ROOT}/data/stage3/loco/{held}/real.h5ad")
    assert [str(g) for g in R.var_names] == genes, "gene axis differs"
    lab = R.obs["target_gene"].astype(str).values; ctrl = lab == "non-targeting"
    X = sp.csr_matrix(R.X).astype(np.float32); del R
    Xc = X[np.flatnonzero(ctrl)]
    C = ad.AnnData(Xc, obs=pd.DataFrame({"target_gene": "non-targeting", "context": "H"}, index=[f"c{i}" for i in range(Xc.shape[0])]),
                   var=pd.DataFrame(index=genes))
    C.write_h5ad(f"{OUT}/context_H.h5ad")
    train = [c for c in B.CONTEXTS if c not in B.FOLDS[SETS[held]]]
    have = set().union(*(set(B.data()[t]["perts"]) for t in train if not B.is_extra(t)))
    n = pd.Series(lab[~ctrl]).value_counts()
    rng = np.random.default_rng(0)
    targets = sorted(t for t, c in n.items() if c >= 50 and t in have and t in set(genes))
    targets = list(rng.permutation(targets)); parts = np.array_split(np.array(targets), 4)
    for k, part in enumerate(parts):
        d = f"{OUT}/s{k}"; os.makedirs(d, exist_ok=True)
        pd.DataFrame({"target_gene": part}).to_csv(f"{d}/pert_counts.csv", index=False)
        shutil.copy(f"{ROOT}/data/vcc2026/gene_names.csv", f"{d}/gene_names.csv")
        json.dump({"season": "mock", "partition": "mock", "panel_id": f"mock-{held.lower()}-s{k}", "contexts": ["H"], "pert_col": "target_gene",
                   "context_col": "context", "control_label": "non-targeting", "n_genes": len(genes), "cells_per_pert": 400}, open(f"{d}/manifest.json", "w"))
        if os.path.lexists(f"{d}/context_H.h5ad"): os.remove(f"{d}/context_H.h5ad")
        os.symlink(f"{OUT}/context_H.h5ad", f"{d}/context_H.h5ad")
        rows = []
        for t in part:
            r = np.flatnonzero(lab == t)
            rows.append(np.sort(rng.choice(r, 400, replace=False)) if len(r) > 400 else r)
        rows = np.sort(np.concatenate(rows))
        Rk = ad.AnnData(sp.vstack([Xc, X[rows]]).tocsr(), obs=pd.DataFrame({"target_gene": np.concatenate([np.full(Xc.shape[0], "non-targeting"), lab[rows]])},
                        index=[f"r{i}" for i in range(Xc.shape[0] + len(rows))]), var=pd.DataFrame(index=genes))
        Rk.write_h5ad(f"{d}/real.h5ad")
        m = pd.Series(lab[rows]).value_counts()
        print(f"{held} s{k}: {len(part)} targets, truth cells per target median {int(m.median())} (min {m.min()}), {Rk.n_obs:,} rows", flush=True)
