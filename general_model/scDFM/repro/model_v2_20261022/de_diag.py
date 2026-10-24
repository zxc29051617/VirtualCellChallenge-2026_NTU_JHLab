"""Where do the DE members lose? Diagnosis on one mock subset from the scorer's own DE tables (2026-10-23).

    .venv/bin/python -I repro/model_v2_20261022/de_diag.py <variant> [subset]        (from scDFM/)

Reads mock/mock_<variant>_s<k>_ce2/de_{real,pred}.parquet (mock_exam.py with WRITE_DE=1) and the variant's effect
files (cells input, totals input). Prints per-target set sizes (reference N_conf vs our calls), the sign accuracy of
our calls against the real log2 fold changes, and how well each predicted quantity ranks the reference-significant genes.
"""
import os, sys
import numpy as np, pandas as pd, polars as pl

HERE = os.path.dirname(os.path.abspath(__file__)); EFF = "/home/zxc29051617/VCC/data/stage3/effects"
v = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = f"{HERE}/mock/mock_{v}_s{k}_ce2"
R = pl.read_parquet(f"{d}/de_real.parquet").to_pandas(); P = pl.read_parquet(f"{d}/de_pred.parquet").to_pandas()
print("columns:", list(R.columns))
key = ["target", "feature"]
M = R[key + ["log2_fold_change", "p_value", "p_adj"]].merge(P[key + ["log2_fold_change", "p_value", "p_adj"]], on=key, suffixes=("_r", "_p"))
M = M[M.target != M.feature]                                     # the scorer drops each target's own gene
print(f"{M.target.nunique()} targets, {len(M):,} (target, gene) pairs")

rows = []
for t, g in M.groupby("target"):
    rs = g.p_adj_r < 0.05; ps = g.p_adj_p < 0.05
    defined = np.isfinite(g.log2_fold_change_r) & (g.log2_fold_change_r != 0)
    match = defined & (np.sign(g.log2_fold_change_p) == np.sign(g.log2_fold_change_r))
    k_ = int((ps & defined & match).sum()); n_pred = int((ps & defined).sum()); n_conf = int(rs.sum())
    tp = int((rs & ps).sum()); union = int(rs.sum() + ps.sum() - tp)
    # sign accuracy on the reference-significant genes, by our p-value order (reach's pool)
    gr = g[rs].sort_values(["p_value_p"]); acc_top = np.mean(np.sign(gr.log2_fold_change_p.values[:20]) == np.sign(gr.log2_fold_change_r.values[:20])) if len(gr) else np.nan
    rows.append(dict(target=t, n_conf=n_conf, n_pred=n_pred, k=k_, prec=k_ / max(n_pred, 1), fid=k_ / max(n_pred, n_conf, 1) if (n_pred or n_conf) else 1.0,
                     tp=tp, jac=tp / union if union else 1.0, sigacc_R=np.mean(np.sign(gr.log2_fold_change_p) == np.sign(gr.log2_fold_change_r)) if len(gr) else np.nan,
                     acc_top20=acc_top))
T = pd.DataFrame(rows).set_index("target")
pd.set_option("display.width", 200)
print(T.describe().round(3).to_string())
print("\nn_conf vs n_pred (quartiles of n_conf):")
T["q"] = pd.qcut(T.n_conf.rank(method="first"), 4, labels=False)
print(T.groupby("q")[["n_conf", "n_pred", "prec", "fid", "jac", "sigacc_R", "acc_top20"]].mean().round(3).to_string())
print(f"\ncorr(log n_conf, log n_pred) = {np.corrcoef(np.log1p(T.n_conf), np.log1p(T.n_pred))[0, 1]:.3f}")
