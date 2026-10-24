"""Process the downloaded KOLF2.1J raw-count region (tools/fetch_kolf.sh) without HDF5:
the chunk map comes from the v1 B-tree leaf nodes ('TREE', type 1, level 0) interleaved with the chunks.

Writes
  data/stage3/stats_ext/KOLF.npz       pseudobulk sums per target gene (+ '__ctrl__'), VCC gene axis (vcc/stats.py format)
  data/stage3/kolf_KOLF.h5ad           single cells of the VCC targets (all) and 20,000 random controls, VCC gene axis

    python kolf_process.py [--scan-only]
"""
import argparse, os, struct, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "/home/zxc29051617/VCC/data"
REGION = f"{ROOT}/external/kolf/counts_region.bin"
META = f"{ROOT}/external/kolf/kolf_meta.npz"
BASE = 94_919_293_556                       # file offset of byte 0 of REGION
DATA0, IND0 = 94_919_293_556, 126_410_487_620
NNZ, NCOL, NROW = 7_872_183_461, 37_567, 2_659_209
CH_DATA, CH_IND = 120_120, 60_060           # elements per chunk (480,480 bytes each)

p = argparse.ArgumentParser(); p.add_argument("--scan-only", action="store_true"); a = p.parse_args()
R = np.memmap(REGION, dtype=np.uint8, mode="r")
t0 = time.time()


def scan_btree_leaves():
    """Find every v1 B-tree leaf of chunked raw data in the region; return {chunk coord: file address} per dataset."""
    cache = f"{ROOT}/external/kolf/chunkmap.npz"
    if os.path.exists(cache):
        z = np.load(cache); return z["data"], z["ind"]
    n_data = (NNZ + CH_DATA - 1) // CH_DATA; n_ind = (NNZ + CH_IND - 1) // CH_IND
    addr_data = np.full(n_data, -1, np.int64); addr_ind = np.full(n_ind, -1, np.int64)
    step = 256 << 20; nleaf = 0
    for lo in range(0, len(R), step):
        buf = bytes(R[lo:min(len(R), lo + step + 4096)])
        k = buf.find(b"TREE")
        while 0 <= k < step:
            typ, lev, used = buf[k + 4], buf[k + 5], struct.unpack_from("<H", buf, k + 6)[0]
            if typ == 1 and lev == 0 and 0 < used <= 64:
                off = k + 24
                for e in range(used):
                    csize, mask, c0, c1 = struct.unpack_from("<IIqq", buf, off); child = struct.unpack_from("<q", buf, off + 24)[0]
                    off += 32
                    if DATA0 <= child < IND0 and c0 % CH_DATA == 0 and csize == CH_DATA * 4:
                        addr_data[c0 // CH_DATA] = child
                    elif child >= IND0 and c0 % CH_IND == 0 and csize == CH_IND * 8:
                        addr_ind[c0 // CH_IND] = child
                nleaf += 1
            k = buf.find(b"TREE", k + 4)
        print(f"  scanned {min(len(R), lo + step) / 2**30:.1f} GiB, {nleaf} leaves, "
              f"data {np.mean(addr_data >= 0):.1%} ind {np.mean(addr_ind >= 0):.1%}  {time.time() - t0:.0f}s", flush=True)
    np.savez(cache, data=addr_data, ind=addr_ind)
    return addr_data, addr_ind


addr_data, addr_ind = scan_btree_leaves()
print(f"chunk map: data {np.sum(addr_data >= 0)}/{len(addr_data)}, indices {np.sum(addr_ind >= 0)}/{len(addr_ind)}", flush=True)
assert (addr_data >= 0).all() and (addr_ind >= 0).all(), "incomplete chunk map"
if a.scan_only: sys.exit()


def elements(addr, ch, itemsize, dtype, lo, hi):
    """Elements [lo, hi) of a chunked 1-D dataset, read from REGION via the chunk map."""
    out = np.empty(hi - lo, dtype); pos = lo
    while pos < hi:
        c = pos // ch; s = pos - c * ch; e = min(ch, hi - c * ch)
        b0 = addr[c] - BASE + s * itemsize
        out[pos - lo:pos - lo + (e - s)] = np.frombuffer(R[b0:b0 + (e - s) * itemsize], dtype)
        pos += e - s
    return out


z = np.load(META, allow_pickle=True)
indptr = z["indptr"]; assert len(indptr) == NCOL + 1 and indptr[-1] == NNZ
target = z["gene_target__cats"][z["gene_target__codes"]] if "gene_target__codes" in z.files else z["gene_target"]
target = np.asarray(target).astype(str)
ctrl_names = {t for t in set(target) if t.lower() in ("non-targeting", "nontargeting", "ntc", "non_targeting", "control", "safe-targeting")}
print("control labels:", ctrl_names, flush=True)
grp = np.where(np.isin(target, list(ctrl_names)), "__ctrl__", target)
names = np.array(sorted(set(grp))); gid = np.searchsorted(names, grp).astype(np.int32)
genes = pd.read_csv(f"{ROOT}/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); gi = {g: i for i, g in enumerate(genes)}
var = z["var_index"].astype(str)
col = np.array([gi.get(v, -1) for v in var])
print(f"{len(names) - 1} targets, {np.sum(grp == '__ctrl__'):,} control cells, {np.sum(col >= 0)} of {NCOL} genes on the VCC axis", flush=True)

vcc = set(pd.read_csv(f"{ROOT}/vcc2026/pert_counts.csv").target_gene.astype(str))
rng = np.random.default_rng(0)
ctrl_idx = np.flatnonzero(grp == "__ctrl__"); ctrl_pick = rng.choice(ctrl_idx, min(20000, len(ctrl_idx)), replace=False)
sel = np.zeros(NROW, bool); sel[np.isin(target, list(vcc))] = True; sel[ctrl_pick] = True
newrow = np.full(NROW, -1, np.int64); newrow[sel] = np.arange(sel.sum())
print(f"single cells kept: {sel.sum():,}", flush=True)

sums = np.zeros((len(names), len(genes)), np.float64)
rows_o, cols_o, vals_o = [], [], []
B = 256
for j0 in range(0, NCOL, B):
    j1 = min(NCOL, j0 + B); lo, hi = indptr[j0], indptr[j1]
    rows = elements(addr_ind, CH_IND, 8, np.int64, lo, hi)
    vals = elements(addr_data, CH_DATA, 4, np.float32, lo, hi)
    cj = np.repeat(np.arange(j0, j1), np.diff(indptr[j0:j1 + 1]))
    vc = col[cj]; ok = vc >= 0
    if ok.any():
        S = np.bincount(gid[rows[ok]].astype(np.int64) * (j1 - j0) + (cj[ok] - j0), weights=vals[ok],
                        minlength=len(names) * (j1 - j0)).reshape(len(names), j1 - j0)
        keep = np.flatnonzero(col[j0:j1] >= 0)
        np.add.at(sums.T, col[j0:j1][keep], S[:, keep].T)
        m = ok & sel[rows]
        rows_o.append(newrow[rows[m]].astype(np.int32)); cols_o.append(vc[m].astype(np.int32)); vals_o.append(vals[m])
    if (j0 // B) % 10 == 0:
        print(f"  genes {j1:,}/{NCOL:,}  elements {hi:,}/{NNZ:,}  {time.time() - t0:.0f}s", flush=True)

ncell = np.bincount(gid, minlength=len(names))
measured = np.zeros(len(genes), bool); measured[col[col >= 0]] = True
os.makedirs(f"{ROOT}/stage3/stats_ext", exist_ok=True)
np.savez_compressed(f"{ROOT}/stage3/stats_ext/KOLF.npz", names=names, sums=sums.astype(np.float32), ncell=ncell,
                    measured=measured, genes=np.array(genes))
print("wrote KOLF stats", flush=True)

import anndata as ad
X = sp.csr_matrix((np.concatenate(vals_o), (np.concatenate(rows_o), np.concatenate(cols_o))), shape=(int(sel.sum()), len(genes)))
X.sum_duplicates(); X.eliminate_zeros()
keep_rows = np.flatnonzero(sel)
obs = pd.DataFrame({"cell_line": "KOLF", "target_gene": np.where(grp[keep_rows] == "__ctrl__", "Non-Targeting", target[keep_rows]),
                    "role": np.where(grp[keep_rows] == "__ctrl__", "control", "vcc_target")},
                   index=z["obs_index"][keep_rows].astype(str) if "obs_index" in z.files else keep_rows.astype(str))
for c in ("cell_line", "target_gene", "role"): obs[c] = obs[c].astype("category")
obs["is_control"] = obs.role == "control"; obs["total_counts"] = np.asarray(X.sum(1)).ravel()
A = ad.AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame({"measured": measured}, index=genes))
A.write_h5ad(f"{ROOT}/stage3/kolf_KOLF.h5ad")
print(f"wrote {ROOT}/stage3/kolf_KOLF.h5ad: {A.shape}, nnz {X.nnz:,}, {time.time() - t0:.0f}s", flush=True)
