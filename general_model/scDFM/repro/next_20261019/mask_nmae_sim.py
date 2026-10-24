"""Offline (H1/KOLF200 V6 DE tables): nmae if the predicted lfc were kept only on each target's top-N genes by
agreement x |t_eq| (others 0), and the share of the real-DE |lfc| mass inside the mask (2026-10-20)."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import bulk_eval as BE, sigexp as S, weights_screen as W
SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"
for name in ["H1", "KOLF200"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); targets = B["targets"]; ti = {t: i for i, t in enumerate(targets)}; oc = B["oc"]
    R = pd.read_parquet(f"de_sets/{name}/de_real.parquet"); P = pd.read_parquet(f"de_sets/{name}/de_pred.parquet")
    M = R.merge(P, on=["target", "feature"], suffixes=("_r", "_p"))
    teq = W.trans(C, train, targets, 0, W.WEIGHTS["equal"])
    n = np.zeros_like(oc); s1 = np.zeros_like(oc); pos = np.zeros_like(oc)
    for c in train:
        Y = C[c]["Ysrc"]
        for j, t in enumerate(C[c]["targets"]):
            i = ti.get(t)
            if i is None: continue
            y = Y[j]; ok = np.isfinite(y); yz = np.where(ok, y, 0); n[i] += ok; s1[i] += yz; pos[i] += yz > 0
    mean = s1 / np.maximum(n, 1); agree = np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))
    key = np.where(np.abs(oc) > 1e-9, 1e6, np.abs(teq) * agree)
    for N in [300, 1000, 3000]:
        thr = -np.sort(-key, axis=1)[:, N - 1:N]; mask = key >= thr
        nm, cov = [], []
        for t, d in M.groupby("target", sort=False):
            d = d[(d.p_adj_r < 0.05) & (d.feature != t) & np.isfinite(d.log2_fold_change_r) & np.isfinite(d.log2_fold_change_p)]
            if len(d) < 10: continue
            mk = mask[ti[t], [S.GI[f] for f in d.feature.values]]
            lr = d.log2_fold_change_r.values; lp = np.where(mk, d.log2_fold_change_p.values, 0.0)
            nm.append(np.abs(lp - lr).mean() / np.abs(lr).mean()); cov.append(np.abs(lr[mk]).sum() / np.abs(lr).sum())
        print(f"{name} top-{N}: nmae {np.mean(nm):.4f} (dense as scored: see agg), real-DE |lfc| mass inside mask {np.mean(cov):.3f}", flush=True)
