"""Pseudobulk statistics for the extra training lines, over ALL perturbations of each screen (not only the
VCC targets). Same npz format as vcc/stats.py: per perturbation group (target gene, plus '__ctrl__') the
summed raw counts on the VCC gene axis, the number of cells, and which VCC genes the source measured.

    python -m vcc.stats_ext replogle K562ess  data/external/replogle2022/K562_essential_raw_singlecell_01.h5ad
    python -m vcc.stats_ext replogle HepG2    data/external/nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad
    python -m vcc.stats_ext replogle Jurkat   data/external/nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad
    python -m vcc.stats_ext feng     FengNF   data/external/feng2025/GenomeWideScreen_NonFitnessGenes
    python -m vcc.stats_ext feng     FengF    data/external/feng2025/GenomeWideScreen_FitnessGenes
Output: data/stage3/stats_ext/<name>.npz
"""
import gzip, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "/home/zxc29051617/VCC/data"
OUT = f"{ROOT}/stage3/stats_ext"
CTRL = "__ctrl__"


def vcc_genes():
    return pd.read_csv(f"{ROOT}/vcc2026/gene_names.csv").gene_name.astype(str).tolist()


class Accum:
    """Sums raw counts per group on the VCC gene axis from (cells x source genes) chunks. Source genes are
    matched to VCC genes by symbol; several source genes with one symbol are added together."""

    def __init__(self, groups, src_genes):
        self.genes = vcc_genes(); gi = {g: i for i, g in enumerate(self.genes)}
        col = np.array([gi.get(str(g), -1) for g in src_genes]); self.ok = np.flatnonzero(col >= 0)
        self.M = sp.csr_matrix((np.ones(len(self.ok)), (np.arange(len(self.ok)), col[self.ok])),
                               shape=(len(self.ok), len(self.genes)))
        self.measured = np.zeros(len(self.genes), bool); self.measured[col[self.ok]] = True
        self.names = np.array(sorted(set(groups)))
        self.index = {g: i for i, g in enumerate(self.names)}
        self.sums = np.zeros((len(self.names), len(self.genes)), np.float64)
        self.ncell = np.zeros(len(self.names), np.int64)

    def indicator(self, gidx):
        return sp.csr_matrix((np.ones(len(gidx)), (gidx, np.arange(len(gidx)))), shape=(len(self.names), len(gidx)))

    def add(self, X, gidx):
        """X: cells x source genes raw counts (sparse or dense); gidx: group index per cell."""
        X = sp.csr_matrix(X)[:, self.ok]
        self.sums += ((self.indicator(gidx) @ X) @ self.M).toarray()
        self.ncell += np.bincount(gidx, minlength=len(self.names))

    def add_gene_rows(self, R, gidx_all, rows):
        """R: source-gene rows x all cells (dense, a block of a genes x cells matrix); rows: their source indices."""
        keep = np.isin(rows, self.ok)
        if not keep.any(): return
        S = np.asarray(self.indicator(gidx_all) @ R[keep].T)            # groups x kept rows
        vcol = self.M[np.searchsorted(self.ok, rows[keep])].indices      # one VCC column per source row
        np.add.at(self.sums.T, vcol, S.T)

    def save(self, name, **extra):
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(f"{OUT}/{name}.npz", names=self.names, sums=self.sums.astype(np.float32), ncell=self.ncell,
                            measured=self.measured, genes=np.array(self.genes), **extra)
        k = self.index.get(CTRL)
        print(f"{name}: {len(self.names) - 1} perturbation groups, {self.ncell[k] if k is not None else 0} control cells, "
              f"{self.measured.sum()} VCC genes measured -> {OUT}/{name}.npz", flush=True)


def replogle(name, path, chunk=20000):
    """Replogle 2022 / Nadig 2025 format: obs.gene ('non-targeting' = control), var.gene_name, raw counts in X."""
    import anndata as ad
    A = ad.read_h5ad(path, backed="r")
    grp = np.where(A.obs.gene.astype(str).values == "non-targeting", CTRL, A.obs.gene.astype(str).values)
    acc = Accum(grp, A.var.gene_name.astype(str).values)
    gidx = np.array([acc.index[g] for g in grp]); t0 = time.time()
    for lo in range(0, A.n_obs, chunk):
        hi = min(A.n_obs, lo + chunk)
        acc.add(A.X[lo:hi], gidx[lo:hi])
        print(f"  {name}: {hi:,}/{A.n_obs:,} cells {time.time() - t0:.0f}s", flush=True)
    acc.save(name)


def feng(name, prefix, batch=256):
    """Feng 2025 hiPSC: dense genes x cells CSV of UMI counts + cell metadata with Guide_Call '<GENE>_<protospacer>'.
    Very few non-targeting cells were captured, so cells without a guide call ('unassigned', a mixture of all
    perturbations, each mostly without effect) serve as the reference group."""
    meta = pd.read_csv(prefix + "_Cell-Metadata.tsv.gz", sep="\t")
    call = meta.Guide_Call.astype(str).values
    grp = np.where(call == "unassigned", CTRL, np.array([c.split("_", 1)[0] for c in call]))
    by_cell = dict(zip(meta.Cell_ID.str.replace(r"^MP-", "", regex=True), grp))
    f = gzip.open(prefix + "_RNA-UMI-Counts.csv.gz", "rt")
    cells = [c.strip('"') for c in f.readline().rstrip("\n").split(",")[1:]]
    cgrp = np.array([by_cell.get(c, "__nometa__") for c in cells])
    print(f"{name}: {len(cells):,} cells, {np.mean(cgrp == '__nometa__'):.1%} without metadata", flush=True)
    # gene list = first column; read it in one cheap pass so Accum knows the source genes
    src = []; t0 = time.time()
    for line in f:                       # row names look like 'ENSG00000186092:OR4F5:Gene-Expression'
        nm = line[:line.index(",")].strip('"'); src.append(nm.split(":")[1] if nm.count(":") >= 2 else nm)
    f.close()
    acc = Accum(cgrp, src); gidx = np.array([acc.index[g] for g in cgrp])
    acc.ncell[:] = np.bincount(gidx, minlength=len(acc.names))
    print(f"  {len(src):,} genes listed {time.time() - t0:.0f}s", flush=True)
    f = gzip.open(prefix + "_RNA-UMI-Counts.csv.gz", "rt"); f.readline()
    R = np.zeros((batch, len(cells)), np.float32); rows = []; r = 0
    for k, line in enumerate(f):
        R[len(rows)] = np.fromstring(line[line.index(",") + 1:], sep=",", dtype=np.float32); rows.append(k)
        if len(rows) == batch:
            acc.add_gene_rows(R, gidx, np.array(rows)); rows = []
            if k % (batch * 20) < batch: print(f"  {name}: {k + 1:,} genes {time.time() - t0:.0f}s", flush=True)
    if rows: acc.add_gene_rows(R[:len(rows)], gidx, np.array(rows))
    acc.save(name)


if __name__ == "__main__":
    kind, name, path = sys.argv[1:4]
    {"replogle": replogle, "feng": feng}[kind](name, path)
