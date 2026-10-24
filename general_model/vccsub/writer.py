"""Streaming writer for VCC 2026 prediction h5ad files.

A full submission is 360,000 cells x 18,533 genes; held in memory as one CSR matrix it
needs ~8 bytes per stored entry, which at real-cell density (~6,000 per cell) is ~17 GB.
This writes obs/var up front and appends X one perturbation block at a time, so peak
memory is one block regardless of submission size.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import sparse_dataset, write_elem

CONTEXT_COL = "context"
PERT_COL = "target_gene"


def read_gene_names(path: str) -> list[str]:
    genes = pd.read_csv(path)["gene_name"].astype(str).tolist()
    assert len(genes) == len(set(genes)), "duplicate gene names"
    return genes


def read_perts(path: str) -> list[str]:
    return pd.read_csv(path)["target_gene"].astype(str).tolist()


class PredictionWriter:
    """Usage:
        w = PredictionWriter(path, genes, contexts, perts, cells_per_pert=400)
        for ctx in contexts:
            for pert in perts:
                w.add(ctx, pert, counts)   # (cells_per_pert, n_genes) int-valued
        w.close()
    Blocks must arrive in exactly the (context, perturbation) order given.
    """

    def __init__(self, path, genes, contexts, perts, cells_per_pert=400):
        self.genes, self.contexts, self.perts, self.k = list(genes), list(contexts), list(perts), cells_per_pert
        self.order = [(c, p) for c in self.contexts for p in self.perts]
        self.pos = 0
        self.nnz = 0
        n = len(self.order) * self.k
        obs = pd.DataFrame({
            CONTEXT_COL: pd.Categorical(np.repeat([c for c, _ in self.order], self.k), categories=self.contexts),
            PERT_COL: pd.Categorical(np.repeat([p for _, p in self.order], self.k), categories=self.perts),
        }, index=pd.Index([f"cell_{i:07d}" for i in range(n)]))
        var = pd.DataFrame(index=pd.Index(self.genes))
        self.f = h5py.File(path, "w")
        self.f.attrs["encoding-type"] = "anndata"
        self.f.attrs["encoding-version"] = "0.1.0"
        write_elem(self.f, "obs", obs)
        write_elem(self.f, "var", var)
        for key in ("obsm", "varm", "obsp", "varp", "layers", "uns"):
            write_elem(self.f, key, {})
        self.X = None

    def add(self, context, pert, counts):
        exp = self.order[self.pos]
        if (context, pert) != exp:
            raise ValueError(f"block {self.pos}: got {(context, pert)}, expected {exp}")
        m = sp.csr_matrix(counts, dtype=np.float32)
        if m.shape != (self.k, len(self.genes)):
            raise ValueError(f"block shape {m.shape} != {(self.k, len(self.genes))}")
        m.eliminate_zeros()                       # explicit zeros count against the density cap
        d = m.data
        if d.size and (d.min() < 0 or not np.all(np.isfinite(d)) or np.any(d != np.round(d))):
            raise ValueError(f"{context}/{pert}: values must be finite non-negative whole numbers")
        if m.shape[0] and np.asarray(m.sum(1)).max() > 1e6:
            raise ValueError(f"{context}/{pert}: a cell exceeds 1,000,000 counts")
        if self.X is None:
            write_elem(self.f, "X", m)  # anndata makes the arrays resizable
            self.X = sparse_dataset(self.f["X"])
        else:
            self.X.append(m)
        self.nnz += m.nnz
        self.pos += 1

    def close(self):
        if self.pos != len(self.order):
            raise ValueError(f"only {self.pos}/{len(self.order)} blocks written")
        self.f.close()
        return self.nnz
