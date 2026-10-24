"""Offline (H1/KOLF200 V6 DE tables): nmae with a per-target scale c x (|t_p| / median |t|)^(g-1) on our predicted lfc
(|t_p| = norm of the model's depth-weighted trans for target p, training sources only) (2026-10-20)."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import bulk_eval as BE, sigexp as S, weights_screen as W, render_eval as E
SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"
for name in ["H1", "KOLF200"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); targets = B["targets"]; ti = {t: i for i, t in enumerate(targets)}
    td = W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]); t5 = E.remove_shared(td.astype(np.float32)).astype(np.float64)
    nr = np.linalg.norm(t5, axis=1); nr = nr / np.median(nr)
    R = pd.read_parquet(f"de_sets/{name}/de_real.parquet"); P = pd.read_parquet(f"de_sets/{name}/de_pred.parquet")
    M = R.merge(P, on=["target", "feature"], suffixes=("_r", "_p"))
    G = []
    for t, d in M.groupby("target", sort=False):
        d = d[(d.p_adj_r < 0.05) & (d.feature != t) & np.isfinite(d.log2_fold_change_r) & np.isfinite(d.log2_fold_change_p)]
        if len(d) >= 10: G.append((nr[ti[t]], d.log2_fold_change_r.values, d.log2_fold_change_p.values))
    out = []
    for g in [1, 2, 3]:
        for c in [1.0, 1.5, 2.0]:
            out.append(f"g{g} c{c}: {np.mean([np.abs(c * n ** (g - 1) * p - r).mean() / np.abs(r).mean() for n, r, p in G]):.4f}")
    print(name, " | ".join(out), flush=True)
