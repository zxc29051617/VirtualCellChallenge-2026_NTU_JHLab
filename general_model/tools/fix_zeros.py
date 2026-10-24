"""Drop explicit zero entries from an h5ad's X in place (cell-eval2 refuses them)."""
import sys, anndata as ad
p = sys.argv[1]; A = ad.read_h5ad(p); n0 = A.X.nnz; A.X.eliminate_zeros()
print(f"{p}: nnz {n0:,} -> {A.X.nnz:,}"); A.write_h5ad(p)
