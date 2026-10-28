"""IGVF TeloHAEC genome-wide CRISPRi CC-Perturb-seq (Engreitz lab, released 2026-10-12) as a source stats file (2026-10-29).

    .venv/bin/python -I ../tools/igvf_telohaec_stats.py <resting h5ad> [<resting h5ad> ...]        (from scDFM/)

Input: the per-replicate resting files of analysis set IGVFDS8166LJMC (IGVFFI0721PXFB, IGVFFI2522NCFR; cells x genes in X,
cells x guides in obsm/guide_assignment, guide names in uns/guide_names). High MOI (~4.8 guides per cell), so there are
almost no cells with control guides only: a perturbation group is every cell carrying at least one guide for the gene
(guide names <seq>_<GENE>_<library>_guide<k>), and the control group '__ctrl__' is every cell of the file (the population;
each gene's own cells are ~0.05 % of it). Groups: the 300 final targets and the 1,000 auxiliary genes (data/coverage), the
same set as the 'vcc'-kind sources. X is read in row blocks; counts are summed on the VCC gene axis (gene_name, unmatched
columns dropped). Replicates are added. Writes data/stage3/stats_ext/TeloHAEC.npz (names, sums, ncell, measured, genes),
the format vcc.contexts reads; the pipeline's 'TeloHAEC' slot picks it up when the file exists.
"""
import re, sys, time
import numpy as np, pandas as pd, h5py, scipy.sparse as sp

OUT = "/home/zxc29051617/VCC/data/stage3/stats_ext/TeloHAEC.npz"; BLOCK = 30000
genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); gi = {g: i for i, g in enumerate(genes)}
targets = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026_final/pert_counts.csv")["target_gene"].astype(str).tolist()
aux = [l.strip() for l in open("/home/zxc29051617/VCC/data/coverage/aux_genes.txt") if l.strip()]
want = list(dict.fromkeys(targets + aux)); wi = {g: k for k, g in enumerate(want)}
PAT = re.compile(r"^[ACGTN]+_(.+)_[A-Za-z]+_guide\d+$")
dec = lambda a: np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])
sums = np.zeros((len(want) + 1, len(genes))); ncell = np.zeros(len(want) + 1, np.int64); measured = np.zeros(len(genes), bool)
t0 = time.time()
for path in sys.argv[1:]:
    H = h5py.File(path, "r")
    vname = dec(H["var/gene_name"][:] if "categories" not in H["var/gene_name"] else
                H["var/gene_name/categories"][:][H["var/gene_name/codes"][:]])
    col = np.array([gi.get(g, -1) for g in vname]); ok = np.flatnonzero(col >= 0); measured[col[ok]] = True
    gnames = dec(H["uns/guide_names"][:]); ggene = np.array([m.group(1) if (m := PAT.match(n)) else "" for n in gnames])
    gcol = np.array([wi.get(g, -1) for g in ggene])
    ga = H["obsm/guide_assignment"]; gptr = ga["indptr"][:]; gidx = ga["indices"][:]
    n = len(gptr) - 1
    cell_of = np.repeat(np.arange(n), np.diff(gptr)); keep = gcol[gidx] >= 0
    G = sp.csr_matrix((np.ones(int(keep.sum())), (cell_of[keep], gcol[gidx][keep])), shape=(n, len(want)))
    G.data[:] = 1.0; G.sum_duplicates(); G.data[:] = 1.0                      # a cell counts once per gene
    ncell[:-1] += np.asarray(G.sum(0)).ravel().astype(np.int64); ncell[-1] += n
    X, ptr = H["X"], H["X/indptr"][:]
    print(f"{path.split('/')[-1]}: {n} cells, {len(vname)} genes ({len(ok)} on the VCC axis), {len(gnames)} guides, "
          f"{int(keep.sum())} assignments to the {len(want)} wanted genes ({time.time() - t0:.0f}s)", flush=True)
    for lo in range(0, n, BLOCK):
        hi = min(n, lo + BLOCK); a, b = ptr[lo], ptr[hi]
        blk = sp.csr_matrix((X["data"][a:b], X["indices"][a:b], ptr[lo:hi + 1] - a), shape=(hi - lo, len(vname)))[:, ok]
        if lo == 0: print(f"  first block: max {blk.max():.1f}, integers {np.allclose(blk.data[:100000], np.round(blk.data[:100000]))}", flush=True)
        sums[:-1, col[ok]] += np.asarray((G[lo:hi].T @ blk).todense())
        sums[-1, col[ok]] += np.asarray(blk.sum(0)).ravel()
        if (lo // BLOCK) % 10 == 0: print(f"  {hi}/{n} cells ({time.time() - t0:.0f}s)", flush=True)
    H.close()
names = np.array(want + ["__ctrl__"])
np.savez(OUT, names=names, sums=sums.astype(np.float32), ncell=ncell, measured=measured, genes=np.array(genes, dtype=object))
print(f"wrote {OUT}: {int((ncell[:-1] >= 20).sum())} groups with >= 20 cells; test genes with >= 50 cells: "
      f"{sum(ncell[wi[t]] >= 50 for t in targets if t in wi)}/{len(targets)}; control cells {ncell[-1]}; "
      f"VCC genes measured {int(measured.sum())} ({time.time() - t0:.0f}s)")
