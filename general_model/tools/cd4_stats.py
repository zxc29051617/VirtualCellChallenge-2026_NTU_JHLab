"""Marson 2025 genome-wide CRISPRi in primary CD4+ T cells -> per-condition stats files in the
format of scDFM/vcc/stats.py (names, sums on the VCC gene axis, ncell, measured, genes).

Run where the 44.6 GB pseudobulk file is local (a cloud box in AWS):
    python cd4_stats.py GWCD4i.pseudobulk_merged.h5ad gene_names.csv targets.txt outdir/
Rows used: targeting pseudobulks with keep_for_DE (min cells, effective guide, total counts) for the
listed targets, and all non-targeting pseudobulks; summed over guides and donors per condition.
"""
import sys, time
import h5py, numpy as np, pandas as pd
from anndata.io import read_elem

src, gene_file, target_file, out = sys.argv[1:5]
genes = pd.read_csv(gene_file).gene_name.astype(str).tolist(); gidx = {g: i for i, g in enumerate(genes)}
targets = set(open(target_file).read().split())
f = h5py.File(src, "r")
obs = read_elem(f["obs"]); var = read_elem(f["var"])
col = np.array([gidx.get(g, -1) for g in var.gene_name.astype(str)])
measured = np.zeros(len(genes), bool); measured[col[col >= 0]] = True
gt = obs.guide_type.astype(str).str.lower()
is_ntc = gt.str.contains("non")
name = np.where(is_ntc, "__ctrl__", obs.perturbed_gene_name.astype(str))
use = (is_ntc & obs.keep_min_cells.astype(bool)) | (obs.perturbed_gene_name.astype(str).isin(targets) & obs.keep_for_DE.astype(bool))
rows = np.flatnonzero(use.values)
print(f"{len(rows)} pseudobulk rows: {int(is_ntc[use].sum())} non-targeting, "
      f"{obs.perturbed_gene_name[use & ~is_ntc].nunique()} targets; {measured.sum()} VCC genes measured", flush=True)
indptr = f["X/indptr"][:]; D = f["X/data"]; I = f["X/indices"]
cond = obs.culture_condition.astype(str).values; ncell_row = obs.n_cells.values
acc = {}
t0 = time.time()
for k, r in enumerate(rows):
    lo, hi = indptr[r], indptr[r + 1]
    c = col[I[lo:hi]]; v = D[lo:hi]; ok = c >= 0
    key = (cond[r], name[r])
    if key not in acc:
        acc[key] = [np.zeros(len(genes)), 0]
    np.add.at(acc[key][0], c[ok], v[ok]); acc[key][1] += int(ncell_row[r])
    if k % 2000 == 0:
        print(f"  {k}/{len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
for cnd in sorted(set(c for c, _ in acc)):
    names = sorted(n for c, n in acc if c == cnd)
    sums = np.stack([acc[(cnd, n)][0] for n in names]).astype(np.float32)
    ncell = np.array([acc[(cnd, n)][1] for n in names])
    np.savez_compressed(f"{out}/CD4_{cnd}.npz", names=np.array(names), sums=sums, ncell=ncell,
                        measured=measured, genes=np.array(genes))
    print(f"CD4_{cnd}: {len(names)-1} targets + controls ({ncell[names.index('__ctrl__')]} control cells)")
