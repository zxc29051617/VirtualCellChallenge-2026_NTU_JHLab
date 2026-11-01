"""Offline (H1/KOLF200 V6 DE tables): is nmae limited by magnitudes? nmae if our predicted lfc were multiplied by c
(global), and median ratio |pred| / |real| on the reference-significant genes (2026-10-20)."""
import numpy as np, pandas as pd
for name in ["H1", "KOLF200"]:
    R = pd.read_parquet(f"de_sets/{name}/de_real.parquet"); P = pd.read_parquet(f"de_sets/{name}/de_pred.parquet")
    M = R.merge(P, on=["target", "feature"], suffixes=("_r", "_p"))
    G = []
    for t, d in M.groupby("target", sort=False):
        d = d[(d.p_adj_r < 0.05) & (d.feature != t) & np.isfinite(d.log2_fold_change_r) & np.isfinite(d.log2_fold_change_p)]
        if len(d) >= 10: G.append((d.log2_fold_change_r.values, d.log2_fold_change_p.values))
    ratio = np.median([np.median(np.abs(p)) / np.median(np.abs(r)) for r, p in G])
    acc = np.mean([np.sum(np.abs(r) * (np.sign(p) == np.sign(r))) / np.sum(np.abs(r)) for r, p in G])
    out = [f"{name}: median |pred|/|real| {ratio:.2f}, |lfc|-weighted sign accuracy {acc:.3f}"]
    for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
        out.append(f"x{c}: {np.mean([np.abs(c * p - r).mean() / np.abs(r).mean() for r, p in G]):.4f}")
    print(" | ".join(out))
