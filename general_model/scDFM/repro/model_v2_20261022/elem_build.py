"""ELEM-1200RS (bulk RNA-seq TPM, CRISPR knockout) as extra TRAINING contexts in the model_v2 format (2026-10-23).

    .venv/bin/python -I repro/model_v2_20261022/elem_build.py        (from scDFM/)

Licence: AI training / academic research allowed; any report that makes material use must carry the acknowledgement
"This work used the ELEM Biotech Virtual Cell Perturbation Transcriptomics Dataset, Release [ELEM-1200RS V1.0], provided
under the ELEM 2026 Priority Access Program."

ELEM_HAP1   HAP1 knockout pools (P1-P4); reference = median log TPM of the knockout pools in the same file and library
            batch (HAP1 has almost no wild-type samples; most knockouts change little, so the median ~ an unperturbed cell)
ELEM_lines  knockout cell lines of the other lines; reference = mean log TPM of that line's wild-type samples (lines
            without wild type skipped); several samples of one target are averaged
Response y = ln((TPM + 1) / (ref + 1)) on the VCC gene axis (symbols; duplicate symbols summed), NaN where ref TPM < 1;
minus the context's mean response (as build_data.py). Rough noise var 0.02 + 1 / (ref + 1); ncell set to 200.
The control profile is set to the mean of the 14 single-cell contexts' profiles (bulk TPM is a different scale), so
the model's cell-state input treats these contexts as neutral.
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

D = "/home/zxc29051617/VCC/data/external/elem"; OUT = "/home/zxc29051617/VCC/data/stage3/model_v2"
genes = pd.read_csv("/home/zxc29051617/VCC/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); gi = {g: i for i, g in enumerate(genes)}
meta = pd.read_excel(f"{D}/ELEM-1200RS_sample_metadata_v1.0.xlsx", sheet_name=0)
col = lambda key: [c for c in meta.columns if key.lower() in str(c).lower()][0]
sid, tcol, ccol, scol, bcol = col("Sample ID"), col("target Gene Name"), col("WT Cell Name"), col("Sample type"), col("Batches")
meta = meta.set_index(sid)

samples = []                                    # (file, sample id, line, type, target, batch, tpm on the VCC axis)
for k in range(1, 5):
    A = ad.read_h5ad(f"{D}/ELEM_1200RS_P{k}.h5ad"); X = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
    sym = A.var["gene_symbol"].astype(str).values; colmap = np.array([gi.get(s, -1) for s in sym]); ok = colmap >= 0
    T = np.zeros((X.shape[0], len(genes)), np.float64)
    np.add.at(T.T, colmap[ok], X[:, ok].T.astype(np.float64))            # sum duplicate symbols
    for i, s in enumerate(map(str, A.obs_names)):
        if s not in meta.index: continue
        r = meta.loc[s]
        samples.append((k, s, str(r[ccol]).strip().upper(), str(r[scol]), str(r[tcol]).strip(), str(r[bcol]), T[i]))
print(f"{len(samples)} annotated samples on the VCC axis", flush=True)
lt = lambda x: np.log(x + 1.0)
mean_ctrl = np.mean([B.data()[c]["ctrl"] for c in B.CONTEXTS], 0).astype(np.float32)


def write(name, rows):
    """rows: list of (target, y vector (NaN allowed), ref tpm)"""
    by = {}
    for t, y, ref in rows:
        if t in ("0", "/", "nan", "") or t not in gi: continue
        by.setdefault(t, []).append((y, ref))
    perts = sorted(by); Y = np.full((len(perts), len(genes)), np.nan); V = np.full_like(Y, np.nan)
    for i, t in enumerate(perts):
        ys = np.stack([y for y, _ in by[t]]); Y[i] = np.nanmean(ys, 0)
        ref = np.mean([r for _, r in by[t]], 0); V[i] = (0.02 + 1 / (ref + 1)) / len(by[t])
    cm = np.nanmean(Y, 0); Yc = Y - cm[None]
    own = np.array([Y[i, gi[t]] for i, t in enumerate(perts)], np.float32)
    np.savez(f"{OUT}/{name}.npz", perts=np.array(perts), ncell=np.full(len(perts), 200), Y=Yc.astype(np.float16), var=V.astype(np.float16),
             own_obs=own, ctrl=mean_ctrl, measured=np.isfinite(Y).any(0), cm=np.nan_to_num(cm).astype(np.float32), genes=np.array(genes))
    print(f"{name}: {len(perts)} targets, own-gene ln ratio median {np.nanmedian(own):+.2f}, usable entries {np.isfinite(Y).mean():.2f}", flush=True)


# HAP1: reference = median per (file, library batch) of the knockout pools
hap = [s for s in samples if s[2] == "HAP1" and "KO" in s[3]]
rows = []
for key in sorted(set((s[0], s[5]) for s in hap)):
    grp = [s for s in hap if (s[0], s[5]) == key]
    if len(grp) < 20: grp_ref = hap                            # small batches: all HAP1 pools as reference
    else: grp_ref = grp
    ref = np.expm1(np.median(np.stack([lt(s[6]) for s in grp_ref]), 0))
    for s in grp:
        y = lt(s[6]) - lt(ref); y[ref < 1] = np.nan; rows.append((s[4], y, ref))
write("ELEM_HAP1", rows)

# other lines: knockout cell lines vs the line's wild type
rows = []
for line in sorted(set(s[2] for s in samples if s[2] != "HAP1")):
    wt = [s for s in samples if s[2] == line and "WT" in s[3]]; ko = [s for s in samples if s[2] == line and "KO" in s[3]]
    if not wt or not ko: continue
    ref = np.expm1(np.mean(np.stack([lt(s[6]) for s in wt]), 0))
    for s in ko:
        y = lt(s[6]) - lt(ref); y[ref < 1] = np.nan; rows.append((s[4], y, ref))
write("ELEM_lines", rows)
