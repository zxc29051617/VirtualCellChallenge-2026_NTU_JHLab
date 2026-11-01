"""Add final-panel (D/E/F) targets to the CD4 stats by reading only their rows of the remote Marson pseudobulk (2026-10-22).

    pertrelay/.venv/bin/python -I tools/cd4_add_remote.py <pert_counts.csv> [--check GENE]

The 44.6 GB GWCD4i.pseudobulk_merged.h5ad (public S3) downloads at ~1 MB/s here, but X is CSR with uncompressed
42,464-element chunks, so the rows of ~160 targets (~22 pseudobulk rows each, ~10k nonzeros per row) are a few GB of
chunks fetched in parallel range requests (tools/http_h5.py). Same row rule and sums as tools/cd4_stats.py:
targeting rows with keep_for_DE, summed over guides, donors and runs per culture condition, ncell = sum of n_cells.
New groups are added to data/stage3/stats/CD4_<condition>.npz (controls and every other group unchanged). --check GENE
recomputes an existing group first and stops unless it matches the stats file exactly.
"""
import argparse, sys, time
import h5py, numpy as np, pandas as pd
from anndata.io import read_elem
sys.path.insert(0, "/home/zxc29051617/VCC/tools")
from http_h5 import RangeFile

URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.pseudobulk_merged.h5ad"
STATS = "/home/zxc29051617/VCC/data/stage3/stats"
p = argparse.ArgumentParser(); p.add_argument("perts"); p.add_argument("--check", default="")
p.add_argument("--batch", type=int, default=15, help="targets per prefetch round")
p.add_argument("--local", default="", help="read a downloaded copy instead (tools/par_download.py); no range requests")
a = p.parse_args(); t0 = time.time()
log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def ranges_of(dset):
    """byte ranges of a dataset's storage (contiguous or every chunk)."""
    if dset.chunks is None:
        off = dset.id.get_offset(); return [] if off is None else [(off, off + dset.id.get_storage_size())]
    return [(i.byte_offset, i.byte_offset + i.size) for i in (dset.id.get_chunk_info(k) for k in range(dset.id.get_num_chunks()))]


class _Local:                                              # stands in for RangeFile when the file is on disk
    nbytes = 0; cache = {}
    def prefetch(self, ranges, workers=0): return 0


if a.local:
    R = _Local(); f = h5py.File(a.local, "r")
else:
    R = RangeFile(URL, block=1 << 18); f = h5py.File(R, "r")
    meta = []
    f["obs"].visititems(lambda n, o: meta.extend(ranges_of(o)) if isinstance(o, h5py.Dataset) else None)
    f["var"].visititems(lambda n, o: meta.extend(ranges_of(o)) if isinstance(o, h5py.Dataset) else None)
    meta += ranges_of(f["X/indptr"])
    log(f"prefetching obs / var / indptr: {R.prefetch(meta)} blocks")
cols = ["culture_condition", "guide_type", "keep_for_DE", "perturbed_gene_name", "n_cells"]
obs = pd.DataFrame({c: read_elem(f["obs"][c]) for c in cols}); var_names = read_elem(f["var"]["gene_name"]) if "gene_name" in f["var"] else read_elem(f["var"]["_index"])
indptr = f["X/indptr"][:]; D = f["X/data"]; I = f["X/indices"]; CH = D.chunks[0]
log(f"{len(obs):,} pseudobulk rows; conditions {sorted(obs.culture_condition.astype(str).unique())}; {R.nbytes/1e6:.0f} MB read")

genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); gidx = {g: i for i, g in enumerate(genes)}
col = np.array([gidx.get(g, -1) for g in pd.Index(var_names).astype(str)])
conds = sorted(obs.culture_condition.astype(str).unique())
st = {c: dict(np.load(f"{STATS}/CD4_{c}.npz", allow_pickle=True)) for c in conds}
have = {c: set(map(str, st[c]["names"])) for c in conds}
targets = pd.read_csv(a.perts)["target_gene"].astype(str).tolist()
nt = obs.guide_type.astype(str).str.lower().str.contains("non").values
name = obs.perturbed_gene_name.astype(str).values; keep = obs.keep_for_DE.astype(bool).values & ~nt
cond = obs.culture_condition.astype(str).values; ncell_row = obs.n_cells.values
todo = [t for t in targets if any(t not in have[c] for c in conds) and (keep & (name == t)).any()]
log(f"{len(todo)} final-panel targets to add (absent from the CD4 stats, present in the screen)")


def sums_for(tlist):
    rows = np.flatnonzero(keep & np.isin(name, tlist))
    if not a.local:
        chunks = sorted({k for r in rows for k in range(indptr[r] // CH, max(indptr[r], indptr[r + 1] - 1) // CH + 1)})
        rng = [(i.byte_offset, i.byte_offset + i.size) for ds in (D, I) for i in (ds.id.get_chunk_info_by_coord((k * CH,)) for k in chunks)]
        R.prefetch(rng)
    acc = {}
    for r in rows:
        lo, hi = indptr[r], indptr[r + 1]; c = col[I[lo:hi]]; v = D[lo:hi]; ok = c >= 0
        k = (cond[r], name[r]); acc.setdefault(k, [np.zeros(len(genes)), 0])
        np.add.at(acc[k][0], c[ok], v[ok]); acc[k][1] += int(ncell_row[r])
    return acc, len(rows)


if a.check:
    acc, n = sums_for([a.check])
    for (c, t), (s, nc) in acc.items():
        nm = list(map(str, st[c]["names"])); j = nm.index(t)
        d = float(np.abs(s.astype(np.float32) - st[c]["sums"][j]).max()); log(f"check {c}/{t}: {n} rows, ncell {nc} vs {st[c]['ncell'][j]}, max |diff| {d}")
        if d > 1e-3 * max(1.0, float(st[c]["sums"][j].max())) or nc != st[c]["ncell"][j]: raise SystemExit("check failed: not the same rows / sums")
    R.cache = {}

new = {c: {} for c in conds}
for b in range(0, len(todo), a.batch):
    acc, n = sums_for(todo[b:b + a.batch])
    for (c, t), v in acc.items():
        if t not in have[c]: new[c][t] = v
    R.cache = {}
    log(f"{min(b + a.batch, len(todo))}/{len(todo)} targets, {n} rows this round, {R.nbytes/1e9:.2f} GB read, {time.time()-t0:.0f}s")
for c in conds:
    nm = list(map(str, st[c]["names"])) + list(new[c]); order = np.argsort(nm)
    sums = np.vstack([st[c]["sums"]] + [v[0][None].astype(np.float32) for v in new[c].values()]) if new[c] else st[c]["sums"]
    ncell = np.concatenate([st[c]["ncell"], [v[1] for v in new[c].values()]]) if new[c] else st[c]["ncell"]
    out = dict(st[c]); out.update(names=np.array(nm)[order], sums=sums[order], ncell=np.asarray(ncell)[order])
    np.savez_compressed(f"{STATS}/CD4_{c}.npz", **out)
    log(f"CD4_{c}: {len(new[c])} groups added -> {len(nm) - 1} perturbations; final panel covered {sum(t in set(nm) for t in targets)}/{len(targets)}")
