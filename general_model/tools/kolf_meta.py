"""KOLF obs/var metadata (cell -> target gene, batch, UMI) and the counts indptr, read remotely (writes kolf_meta.npz for tools/kolf_process.py).

    pertrelay/.venv/bin/python -I tools/kolf_meta.py https://ndownloader.figshare.com/files/64650261 data/external/kolf/kolf_meta.npz
"""
import os, sys, time, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remotefile import RemoteFile
url, out = sys.argv[1], sys.argv[2]
t0 = time.time()
f = RemoteFile(url, block=64 << 10, max_blocks=8192, verbose=False)
h = h5py.File(f, "r"); print("opened", flush=True)
obs = {}
def get(path, key):
    g = h[path]
    if isinstance(g, h5py.Group):
        obs[key + "__codes"] = g["codes"][:]
        obs[key + "__cats"] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]])
    else:
        v = g[:]; obs[key] = np.array([x.decode() for x in v]) if v.dtype.kind in "OS" else v
    print(key, "%.0fs" % (time.time() - t0), flush=True)
for c in ["gene_target", "perturbed", "perturbation_type", "batch", "channel", "n_UMI_counts", "n_gRNA", "celltype"]:
    get(f"obs/{c}", c)
get("var/_index", "var_index"); get("var/gene_ids", "var_gene_ids")
obs["indptr"] = h["layers/counts/indptr"][:]; print("indptr", obs["indptr"][:3], obs["indptr"][-1], flush=True)
obs["attrs_counts"] = np.array([str(dict(h["layers/counts"].attrs))])
np.savez_compressed(out, **obs)
print("saved", out, "%.0fs" % (time.time() - t0), flush=True)
