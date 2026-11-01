"""Offline headroom check for sparse calling (2026-10-20, H1/KOLF200 V6 DE tables; analysis only).
Recomputes the scorer's fid_yield_raw = k / max(n_pred, N_conf) and raw reach (prefix purity >= 0.9 over the
reference-significant genes ranked by predicted significance) from the tables, then simulates keeping only the top-K
predicted-significant genes per target (others' p_adj set to 1), K = fixed or = c x N_conf (oracle size, for headroom)."""
import sys
import numpy as np, pandas as pd
A = 0.05
def metrics(M, keep_k=None, kfac=None):
    fids, reaches, jacs = [], [], []
    for t, d in M.groupby("target", sort=False):
        d = d[d.feature != t]
        real_sig = (d.p_adj_r < A).values; ncf = int(real_sig.sum())
        lr = d.log2_fold_change_r.values; lp = d.log2_fold_change_p.values; pp = d.p_adj_p.values.copy(); pv = d.p_value_p.values.copy()
        order = np.lexsort((-np.abs(lp), pv, pp))                       # sort keys: p_adj, p_value, |lfc| desc
        sig_p = pp < A
        if keep_k is not None or kfac is not None:
            K = keep_k if keep_k is not None else int(round(kfac * ncf))
            ranked_sig = [i for i in order if sig_p[i]][:K]; sig_p = np.zeros_like(sig_p); sig_p[ranked_sig] = True
            pp = np.where(sig_p, pp, 1.0); order = np.lexsort((-np.abs(lp), pv, pp, ~sig_p))
        indenom = np.isfinite(lr) & (lr != 0); match = indenom & (np.sign(lp) == np.sign(lr))
        sc = sig_p & indenom; n_pred = int(sc.sum()); k = int((sc & match).sum())
        fids.append(k / max(n_pred, ncf) if max(n_pred, ncf) > 0 else (1.0 if n_pred == 0 else 0.0))
        # reach: adjudicated universe = reference-significant genes, ranked by the prediction's significance
        o = [i for i in order if real_sig[i] and indenom[i]]
        if ncf > 0 and o:
            m = match[o].astype(float); pur = np.cumsum(m) / np.arange(1, len(o) + 1)
            ks = np.flatnonzero(pur >= 0.9); reaches.append((ks[-1] + 1) / ncf if len(ks) else 0.0)
        jacs.append(len(np.flatnonzero(sig_p & real_sig)) / max(len(np.flatnonzero(sig_p | real_sig)), 1))
    return np.mean(fids), np.mean(reaches), np.mean(jacs)
for name in ["H1", "KOLF200"]:
    R = pd.read_parquet(f"de_sets/{name}/de_real.parquet"); P = pd.read_parquet(f"de_sets/{name}/de_pred.parquet")
    M = R.merge(P, on=["target", "feature"], suffixes=("_r", "_p"))
    f, r, j = metrics(M); print(f"{name} as scored (V6 cells): fid_yield_raw {f:.4f} reach {r:.4f} jac {j:.4f}", flush=True)
    for K in [10, 25, 50, 100, 200]:
        f, r, j = metrics(M, keep_k=K); print(f"   keep top-{K:<4d} fid {f:.4f} reach {r:.4f} jac {j:.4f}", flush=True)
    for c in [0.25, 0.5, 1.0]:
        f, r, j = metrics(M, kfac=c); print(f"   keep {c} x N_conf (oracle size) fid {f:.4f} reach {r:.4f} jac {j:.4f}", flush=True)
