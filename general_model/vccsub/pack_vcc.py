"""Package a prediction .h5ad into a .vcc without loading it into memory.

`vcc prep` reads the whole matrix (19.4 GB peak for a full 2.09e9-nnz submission, more than this machine has). For
raw-count input it leaves X untouched and writes: a minimal h5ad (X as CSR float32; obs = target_gene + context with a
0..n-1 index; var = gene names; nothing else), zstd level 3, then a tar of meta.json + pred.h5ad.zst with normalised
headers (vcc/prep.py::_write_vcc). This does the same by streaming: it copies X chunk by chunk into a fresh minimal
h5ad, checks the same invariants prep checks (gene order, whole non-negative counts, 400 cells per target per context,
no controls), compresses and tars.
vcc 0.3.0 (final phase) also records the panel in meta.json: contexts, perts_fingerprint and panel_id (from the
manifest.json next to --perts); a .vcc without contexts uploads but fails at scoring.

Run with the vcc-cli interpreter (it has zstandard and the vcc package):
    ~/.local/share/uv/tools/vcc-cli/bin/python pack_vcc.py out/pred.h5ad out/pred.vcc --genes ../data/vcc2026/gene_names.csv --perts ../data/vcc2026/pert_counts.csv
"""
import argparse, json, os, tarfile, tempfile, time
import h5py, numpy as np, pandas as pd, zstandard as zstd
import anndata as ad
from anndata.io import write_elem
from vcc import __version__ as VCC_VERSION
from vcc.panel import perts_fingerprint

p = argparse.ArgumentParser()
p.add_argument("inp"); p.add_argument("out")
p.add_argument("--genes", required=True); p.add_argument("--perts", required=True)
p.add_argument("--contexts", nargs="+", default=None, help="default: the contexts in manifest.json next to --perts")
p.add_argument("--cells-per-pert", type=int, default=400)
p.add_argument("--no-check-cell-counts", action="store_true")
p.add_argument("--no-verify-targets", action="store_true")
p.add_argument("--tmp", default=None)
a = p.parse_args()
t0 = time.time()
man_path = os.path.join(os.path.dirname(os.path.abspath(a.perts)), "manifest.json")
manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
if a.contexts is None:
    assert manifest.get("contexts"), f"no --contexts and no manifest.json with contexts next to {a.perts}"
    a.contexts = list(manifest["contexts"])
elif manifest.get("contexts"):
    assert set(a.contexts) == set(manifest["contexts"]), f"--contexts {a.contexts} != manifest {manifest['contexts']}"

genes = pd.read_csv(a.genes)["gene_name"].astype(str).tolist() if "gene_name" in open(a.genes).readline() else \
    [l.strip() for l in open(a.genes) if l.strip()]
perts = set(pd.read_csv(a.perts)["target_gene"].astype(str))
src = h5py.File(a.inp, "r")
obs = ad.io.read_elem(src["obs"]); var = ad.io.read_elem(src["var"])
assert list(var.index.astype(str)) == genes, "gene names / order differ from the gene list"
tg = obs["target_gene"].astype(str).to_numpy(); cx = obs["context"].astype(str).to_numpy()
assert not np.isin(tg, ["non-targeting"]).any(), "controls must not be submitted"
assert set(cx) == set(a.contexts), f"contexts {sorted(set(cx))} != {a.contexts}"
for c in a.contexts:
    got = set(tg[cx == c])
    if not a.no_verify_targets: assert got == perts, f"context {c}: {len(got ^ perts)} target labels differ"
    if not a.no_check_cell_counts:
        cnt = pd.Series(tg[cx == c]).value_counts()
        assert (cnt == a.cells_per_pert).all(), f"context {c}: not {a.cells_per_pert} cells for every target"
X = src["X"]; enc = X.attrs["encoding-type"]
assert enc == "csr_matrix", f"X must be CSR, got {enc}"
n_obs, n_vars = (int(v) for v in X.attrs["shape"])
nnz = int(X["data"].shape[0])
print(f"{a.inp}: {n_obs:,} cells x {n_vars:,} genes, nnz {nnz:,}; checks passed ({time.time() - t0:.0f}s)", flush=True)

tmpdir = tempfile.mkdtemp(dir=a.tmp or os.path.dirname(os.path.abspath(a.out)))
h5_path, zst_path, meta_path = (os.path.join(tmpdir, n) for n in ("pred.h5ad", "pred.h5ad.zst", "meta.json"))
dst = h5py.File(h5_path, "w")
dst.attrs["encoding-type"] = "anndata"; dst.attrs["encoding-version"] = "0.1.0"
new_obs = pd.DataFrame({"target_gene": tg, "context": cx}, index=np.arange(n_obs).astype(str))
for col in new_obs.columns: new_obs[col] = new_obs[col].astype("category")   # write_h5ad's strings_to_categoricals
write_elem(dst, "obs", new_obs)
write_elem(dst, "var", pd.DataFrame(index=pd.Index(genes)))
for key in ("obsm", "varm", "obsp", "varp", "layers", "uns"): write_elem(dst, key, {})
g = dst.create_group("X")
g.attrs["encoding-type"] = "csr_matrix"; g.attrs["encoding-version"] = "0.1.0"; g.attrs["shape"] = np.array([n_obs, n_vars])
step = 200_000_000
ptr_dtype = np.int32 if nnz < 2**31 else np.int64          # what anndata/scipy write when it fits
for name, dtype in (("data", np.float32), ("indices", X["indices"].dtype), ("indptr", ptr_dtype)):
    s = X[name]; d = g.create_dataset(name, shape=s.shape, dtype=dtype)
    for lo in range(0, s.shape[0], step):
        v = s[lo:lo + step]
        if name == "data":
            assert v.size == 0 or (v.min() >= 0 and np.all(v == np.round(v))), "counts must be whole and non-negative"
        d[lo:lo + step] = v.astype(dtype, copy=False)
    print(f"  copied X/{name} ({s.shape[0]:,}) {time.time() - t0:.0f}s", flush=True)
dst.close(); src.close()

meta = {"schema": 1, "nnz": nnz, "n_obs": n_obs, "n_vars": n_vars, "cli_version": VCC_VERSION,
        "contexts": list(a.contexts), "perts_fingerprint": perts_fingerprint(a.perts), "panel_id": manifest.get("panel_id")}
meta = {k: v for k, v in meta.items() if v is not None}          # as vcc/prep.py::_write_vcc drops unknown extras
with open(meta_path, "w", encoding="utf-8") as fh: json.dump(meta, fh, sort_keys=True)
cctx = zstd.ZstdCompressor(level=3, threads=min(8, os.cpu_count() or 1))
with open(h5_path, "rb") as s, open(zst_path, "wb") as d: cctx.copy_stream(s, d)
os.remove(h5_path)
print(f"  compressed -> {os.path.getsize(zst_path) / 2**30:.2f} GiB {time.time() - t0:.0f}s", flush=True)

def _normalize(ti):
    ti.uid = ti.gid = 0; ti.uname = ti.gname = ""; ti.mtime = 0; return ti
with tarfile.open(a.out, "w") as tar:
    tar.add(meta_path, arcname="meta.json", filter=_normalize)
    tar.add(zst_path, arcname="pred.h5ad.zst", filter=_normalize)
os.remove(zst_path); os.remove(meta_path); os.rmdir(tmpdir)
print(f"wrote {a.out} ({os.path.getsize(a.out) / 2**30:.2f} GiB) in {time.time() - t0:.0f}s")
