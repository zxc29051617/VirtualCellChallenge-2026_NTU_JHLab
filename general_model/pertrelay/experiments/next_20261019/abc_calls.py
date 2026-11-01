"""How many genes do our cells call significant on context A? (2026-10-20, no truth needed)

    /home/zxc29051617/VCC/evalenv/bin/python -I abc_calls.py

Renders 30 targets of A with the G3c cells (x4.5 x agreement) and with the G3v cells (x3), both with the g3a totals and
added counts in the top 2 % of cells, then runs the scorer's DE (counts -> CPM 1e6 + log1p, pdex Wilcoxon vs the released
controls, BH) and reports the number of significant genes per target (own gene excluded). Compare with the reference's
typical numbers (scorer docs: 73-91 % of val targets have >= 10).
"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, polars as pl
sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
import vcc.render as VR
from vcc.render import render
from pdex import pdex
VR.TOP_FRAC = 0.02
E = "/home/zxc29051617/VCC/data/stage3/effects"
A = ad.read_h5ad("/home/zxc29051617/VCC/data/vcc2026/context_A.h5ad"); C = sp.csr_matrix(A.X).astype(np.float64); genes = list(A.var_names)
lib = np.asarray(C.sum(1)).ravel(); rate = np.asarray(C.sum(0)).ravel() / lib.sum(); ref_mean = np.asarray(C.sum(0)).ravel() / C.shape[0]
tot = np.load(f"{E}/VCC_gm_vip_g3a_A.npz", allow_pickle=True); targets = [str(t) for t in tot["targets"]]
rng = np.random.default_rng(1); idx = rng.choice(len(targets), 30, replace=False)
for lab, f in [("G3v cells x3", "VCC_gm_vip.npz"), ("G3c cells x4.5 x agreement", "VCC_gm_vip_c12.npz")]:
    cl = np.load(f"{E}/{f}", allow_pickle=True)["lfc"].astype(np.float64)
    blocks, labels = [], []
    for i in idx:
        draw = np.random.default_rng([7, int(i)]).choice(C.shape[0], 400, replace=False)
        blocks.append(render(C[draw], lib[draw], rate, ref_mean, cl[i], np.random.default_rng([8, int(i)]), mode="dual",
                             lfc_bulk=tot["lfc"][i].astype(np.float64), fix_seed=int(i)))
        labels += [targets[i]] * 400
    X = sp.vstack(blocks + [sp.csr_matrix(C.astype(np.float32))]).tocsr().astype(np.float32)
    X = sp.diags(1e6 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X; X.data = np.log1p(X.data)
    ad_ = ad.AnnData(X.tocsr(), obs=pd.DataFrame({"g": labels + ["non-targeting"] * C.shape[0]}, index=[f"c{k}" for k in range(X.shape[0])]),
                     var=pd.DataFrame(index=genes))
    fr = pdex(ad_, groupby="g", mode="ref", reference="non-targeting", geometric_mean=True, epsilon=0.0, is_log1p=True, threads=8)
    fr = fr if isinstance(fr, pl.DataFrame) else pl.from_pandas(fr)
    sig = fr.filter((pl.col("fdr") < 0.05) & (pl.col("feature") != pl.col("target"))).group_by("target").len()
    n = np.array(sig["len"].to_list() + [0] * (30 - sig.height))
    print(f"{lab}: significant genes per target median {np.median(n):.0f} (IQR {np.percentile(n, 25):.0f}-{np.percentile(n, 75):.0f}, max {n.max()})", flush=True)
