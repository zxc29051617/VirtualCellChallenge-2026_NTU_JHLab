"""Stage-2 baseline: every perturbation is 400 control cells resampled from its own
context, optionally binomially thinned to a lower depth so that `vcc prep` fits in RAM.

It carries no perturbation information, so it should score about -0.3 (it is not the
context-mean baseline that defines 0). Its job is to prove the submission interface.

    python baseline_control_resample.py --data ../data/vcc2026 --out out/baseline.h5ad --thin 0.33
"""
import argparse
import os
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from writer import PredictionWriter, read_gene_names, read_perts  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--thin", type=float, default=1.0, help="binomial thinning probability per count")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
genes = read_gene_names(f"{a.data}/gene_names.csv")
perts = read_perts(f"{a.data}/pert_counts.csv")
contexts = ["A", "B", "C"]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
w = PredictionWriter(a.out, genes, contexts, perts, cells_per_pert=400)
for c in contexts:
    ctrl = ad.read_h5ad(f"{a.data}/context_{c}.h5ad")
    assert list(ctrl.var_names) == genes, "control gene order differs from gene_names.csv"
    assert (ctrl.obs["context"] == c).all()
    X = sp.csr_matrix(ctrl.X)
    for pert in perts:
        block = X[rng.choice(X.shape[0], 400, replace=False)].tocsr(copy=True)
        if a.thin < 1:
            block.data = rng.binomial(block.data.astype(np.int64), a.thin).astype(np.float32)
        w.add(c, pert, block)
    print(f"context {c}: done, running nnz {w.nnz:,}", flush=True)
    del ctrl, X
nnz = w.close()
print(f"wrote {a.out}: {len(contexts)*len(perts)*400:,} cells, nnz {nnz:,} ({nnz/(len(contexts)*len(perts)*400):.0f}/cell)")
