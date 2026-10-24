"""Stream X-Atlas/Orion parquet batches from Hugging Face and keep only selected cells,
on the VCC 2026 gene axis, as raw counts.
    python extract_xatlas.py HCT116    -> data/stage3/xatlas_HCT116.h5ad
    python extract_xatlas.py HCT116 <keep.csv> <out.h5ad>      (another selection, e.g. the final-panel targets)
Peak memory is one batch (~0.5 GB parquet); disk holds at most two batches at a time.
"""
import os, sys, time, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp, h5py
from anndata.io import sparse_dataset, write_elem

LINE = sys.argv[1]
ROOT = "/home/zxc29051617/VCC/data"
HF = "https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main"
TMP = f"{ROOT}/external/xatlas_tmp"; os.makedirs(TMP, exist_ok=True); os.makedirs(f"{ROOT}/stage3", exist_ok=True)
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/stage3/xatlas_{LINE}.h5ad"

genes = pd.read_csv(f"{ROOT}/vcc2026/gene_names.csv").gene_name.astype(str).tolist()
gidx = {g: i for i, g in enumerate(genes)}
meta = pd.read_parquet(f"{ROOT}/external/meta/xatlas_gene_metadata.parquet")
tok2vcc = np.full(meta.gene_token_id.max() + 1, -1, dtype=np.int64)
for t, g in zip(meta.gene_token_id, meta.gene_name):
    tok2vcc[t] = gidx.get(g, -1)
measured = np.zeros(len(genes), bool); measured[tok2vcc[tok2vcc >= 0]] = True
keep = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/coverage/xatlas_keep_{LINE}.csv").set_index("cell_barcode")

tree = json.load(urllib.request.urlopen(
    "https://huggingface.co/api/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/main/data"))
files = sorted(x["path"] for x in tree if x["path"].startswith(f"data/{LINE}_"))
batches = {os.path.basename(f)[:-8] for f in files}
need = set(keep.index.str.rsplit("-", n=1).str[1])
files = [f for f in files if os.path.basename(f)[:-8] in need]
print(f"{LINE}: {len(files)} batch files hold the {len(keep):,} selected cells", flush=True)

def fetch(path):
    dst = f"{TMP}/{os.path.basename(path)}"
    for a in range(6):
        try:
            urllib.request.urlretrieve(f"{HF}/{path}", dst + ".part"); os.replace(dst + ".part", dst); return dst
        except Exception as e:
            print("retry", path, e, flush=True); time.sleep(5 * (a + 1))
    raise RuntimeError(path)

f = h5py.File(OUT + ".part", "w"); X = None; obs = []; nnz = 0; t0 = time.time()
with ThreadPoolExecutor(2) as ex:
    futs = [ex.submit(fetch, p) for p in files[:2]]
    for i, path in enumerate(files):
        local = futs[i].result()
        if i + 2 < len(files): futs.append(ex.submit(fetch, files[i + 2]))
        t = pq.read_table(local, columns=["cell_barcode", "gene_token_id", "gene_expression", "total_counts", "sample"])
        bc = t.column("cell_barcode").to_numpy(zero_copy_only=False)
        rows = np.flatnonzero(pd.Index(bc).isin(keep.index))
        if len(rows):
            tok = t.column("gene_token_id").take(rows); val = t.column("gene_expression").take(rows)
            lens = np.diff(tok.offsets.to_numpy()) if hasattr(tok, "offsets") else np.array([len(x) for x in tok.to_pylist()])
            tok_flat = tok.combine_chunks().values.to_numpy(); val_flat = val.combine_chunks().values.to_numpy()
            r = np.repeat(np.arange(len(rows)), lens); c = tok2vcc[tok_flat]; ok = c >= 0
            m = sp.csr_matrix((val_flat[ok].astype(np.float32), (r[ok], c[ok])), shape=(len(rows), len(genes)))
            m.sum_duplicates(); m.eliminate_zeros()
            if X is None:
                write_elem(f, "X", m); X = sparse_dataset(f["X"])
            else:
                X.append(m)
            nnz += m.nnz
            k = keep.loc[bc[rows]]
            obs.append(pd.DataFrame({"cell_line": LINE, "target_gene": k.gene_target.values, "role": k.role.values,
                                     "batch": t.column("sample").take(rows).to_numpy(zero_copy_only=False),
                                     "total_counts": t.column("total_counts").take(rows).to_numpy()},
                                    index=bc[rows]))
        os.remove(local)
        if i % 10 == 0 or i == len(files) - 1:
            n = sum(len(o) for o in obs)
            print(f"  {i+1}/{len(files)} batches, {n:,} cells kept, nnz {nnz:,}, {time.time()-t0:.0f}s", flush=True)

obs = pd.concat(obs)
for col in ("cell_line", "target_gene", "role", "batch"): obs[col] = obs[col].astype("category")
obs["is_control"] = obs.role == "control"
var = pd.DataFrame({"measured": measured}, index=pd.Index(genes))
f.attrs["encoding-type"] = "anndata"; f.attrs["encoding-version"] = "0.1.0"
write_elem(f, "obs", obs); write_elem(f, "var", var)
for key in ("obsm", "varm", "obsp", "varp", "layers"): write_elem(f, key, {})
write_elem(f, "uns", {"source": "X-Atlas/Orion (Xaira), CC BY-NC-SA 4.0", "selection": "tools/select_cells.py"})
f.close(); os.replace(OUT + ".part", OUT)
print(f"wrote {OUT}: {len(obs):,} cells (expected {len(keep):,}), nnz {nnz:,}", flush=True)
