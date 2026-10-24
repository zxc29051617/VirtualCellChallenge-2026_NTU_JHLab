"""Pseudobulk statistics for VIPerturb-seq K562 (Flex, genome_wide_filtered.rds) by streaming the RDS twice.

Pass 1 (skeleton pickle from tools/rds_stream.skeleton) gives column pointers, feature names and meta.data.
Pass 2 streams the RNA counts: `i` (row indices, 1.7e9 int32) is spooled to disk, then `x` is aggregated chunk by chunk
into per-target sums on the VCC gene axis. Control label: 'NO-TARGET'.
Writes data/stage3/stats_ext/VIPK562.npz in vcc/stats.py format.

    rdsenv/bin/python -I tools/viperturb_stats.py
"""
import os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rds_stream as R

RDS = "/home/zxc29051617/VCC/data/external/viperturb/genome_wide_filtered.rds"
SKEL = "/home/zxc29051617/.claude/jobs/3cf1662b/tmp/vip_skel.pkl"
SPOOL = "/home/zxc29051617/VCC/data/external/viperturb/rna_i.int32"
OUT = "/home/zxc29051617/VCC/data/stage3/stats_ext/VIPK562.npz"
t0 = time.time()
o = pickle.load(open(SKEL, "rb"))
md = o.attr["meta.data"]; cols = dict(zip(md.attr["names"].value, md.value))
gene = np.array(cols["gene"].value)
rna = o.attr["assays"].value[0]; cnt = rna.attr["layers"].value[0]
p = np.asarray(cnt.attr["p"].value, np.int64); n_rows, n_cols = (int(v) for v in cnt.attr["Dim"].value)
feat = np.array(rna.attr["features"].attr["dimnames"].value[0].value)
assert len(feat) == n_rows and len(gene) == n_cols and len(p) == n_cols + 1
grp = np.where(gene == "NO-TARGET", "__ctrl__", gene)
names = np.array(sorted(set(grp))); gidx = np.searchsorted(names, grp).astype(np.int64)
NNZ = int(p[-1]); print(f"{n_cols:,} cells, {n_rows:,} features, nnz {NNZ:,}, {len(names) - 1} targets", flush=True)

sums = np.zeros((len(names), n_rows), np.float64)
state = {"i_written": 0, "x_done": 0, "spool": None, "imap": None}


def cb(k, typ, arr):
    if k == 0:                                     # RNA counts @i
        if state["spool"] is None: state["spool"] = open(SPOOL, "wb")
        state["spool"].write(arr.astype("<i4").tobytes()); state["i_written"] += len(arr)
        if state["i_written"] == NNZ:
            state["spool"].close(); state["imap"] = np.memmap(SPOOL, dtype="<i4", mode="r")
            print(f"  spooled i ({NNZ:,}) {time.time() - t0:.0f}s", flush=True)
    elif k == 1:                                   # RNA counts @x
        lo = state["x_done"]; hi = lo + len(arr)
        rows = np.asarray(state["imap"][lo:hi], np.int64)
        col = np.searchsorted(p, np.arange(lo, hi), side="right") - 1
        sums.ravel()[:] += np.bincount(gidx[col] * n_rows + rows, weights=arr.astype(np.float64),
                                       minlength=len(names) * n_rows)
        state["x_done"] = hi
        if (hi // len(arr)) % 25 == 0: print(f"  x {hi:,}/{NNZ:,} {time.time() - t0:.0f}s", flush=True)
        if hi == NNZ: raise StopIteration


try:
    R.stream(RDS, cb)
except StopIteration:
    pass
assert state["x_done"] == NNZ, state["x_done"]
genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist()
gi = {g: i for i, g in enumerate(genes)}; colmap = np.array([gi.get(f, -1) for f in feat])
out = np.zeros((len(names), len(genes)), np.float32); ok = colmap >= 0
np.add.at(out.T, colmap[ok], sums[:, ok].T.astype(np.float32))
measured = np.zeros(len(genes), bool); measured[colmap[ok]] = True
ncell = np.bincount(gidx, minlength=len(names))
np.savez_compressed(OUT, names=names, sums=out, ncell=ncell, measured=measured, genes=np.array(genes))
os.remove(SPOOL)
print(f"wrote {OUT}: {len(names) - 1} targets, {int(ncell[list(names).index('__ctrl__')])} controls, "
      f"{measured.sum()} VCC genes measured, {time.time() - t0:.0f}s", flush=True)
