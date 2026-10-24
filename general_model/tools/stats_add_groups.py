"""Add perturbation groups for a new target list to an existing source stats file (2026-10-22, final panel D/E/F).

The 'vcc'-kind sources (K562 gwps, X-Atlas HCT116/HEK293T, CD4) were extracted for the A/B/C targets plus auxiliary genes
only; the final panel has 171 new targets. This adds them (or replaces a group extracted with fewer cells, e.g. a 64-cell
auxiliary gene) and keeps every other group and the controls ('__ctrl__') untouched.

    python stats_add_groups.py k562bulk <stats.npz> <K562_gwps_raw_bulk_01.h5ad> <pert_counts.csv>
    python stats_add_groups.py cells    <stats.npz> <line_extract.h5ad> <pert_counts.csv>

k562bulk: sums = per-guide-group mean raw UMI x num_cells_filtered, summed over a gene's groups (equals the sum over the
          single-cell file's filtered cells; checked on CTSV: ratio 1.0000, corr 0.99995).
cells   : sums over the extract's cells per target_gene (non-control rows), VCC gene axis.
A group is replaced only when the new one has more cells. Writes the stats file in place (back it up first).
"""
import sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

mode, stats_path, src, perts_path = sys.argv[1:5]
z = np.load(stats_path, allow_pickle=True); st = {k: z[k] for k in z.files}
names = [str(n) for n in st["names"]]; genes = [str(g) for g in st["genes"]]; gi = {g: i for i, g in enumerate(genes)}
targets = pd.read_csv(perts_path)["target_gene"].astype(str).tolist()
new = {}                                                   # target -> (sums on the axis, ncell)
if mode == "k562bulk":
    B = ad.read_h5ad(src)
    X = B.X.toarray() if sp.issparse(B.X) else np.asarray(B.X)
    col = np.array([gi.get(g, -1) for g in B.var["gene_name"].astype(str)]); ok = col >= 0
    gene_of = B.obs_names.str.split("_").str[1].values; ncf = B.obs["num_cells_filtered"].values
    for t in targets:
        rows = np.flatnonzero((gene_of == t) & np.isfinite(ncf))
        if not len(rows): continue
        s = np.zeros(len(genes)); s[col[ok]] = (X[rows][:, ok].astype(np.float64) * ncf[rows, None]).sum(0)
        new[t] = (s, int(round(ncf[rows].sum())))
elif mode == "cells":
    A = ad.read_h5ad(src, backed="r"); assert [str(g) for g in A.var_names] == genes, "gene axis differs"
    lab = A.obs["target_gene"].astype(str).values; ctrl = A.obs["is_control"].values.astype(bool) if "is_control" in A.obs else np.zeros(len(lab), bool)
    want = set(targets)
    for t in sorted(set(lab[~ctrl]) & want):
        rows = np.flatnonzero((lab == t) & ~ctrl)
        X = sp.csr_matrix(A.X[rows[0]:rows[-1] + 1])[rows - rows[0]] if rows[-1] - rows[0] < 50000 else sp.vstack([sp.csr_matrix(A.X[r:r + 1]) for r in rows])
        new[t] = (np.asarray(X.sum(0), dtype=np.float64).ravel(), len(rows))
else:
    raise SystemExit(f"unknown mode {mode}")
added = replaced = kept = 0
sums = list(st["sums"].astype(np.float64)); ncell = list(st["ncell"])
for t, (s, n) in new.items():
    if t in names:
        k = names.index(t)
        if n > ncell[k]: sums[k] = s; ncell[k] = n; replaced += 1
        else: kept += 1
    else:
        names.append(t); sums.append(s); ncell.append(n); added += 1
order = np.argsort(names)                                  # vcc/stats.py writes the groups sorted
st["names"] = np.array(names)[order]; st["sums"] = np.array(sums, np.float32)[order]; st["ncell"] = np.array(ncell)[order]
np.savez_compressed(stats_path, **st)
cover = sum(t in set(names) for t in targets)
print(f"{stats_path}: {added} groups added, {replaced} replaced (more cells), {kept} kept; final panel covered {cover}/{len(targets)}")
