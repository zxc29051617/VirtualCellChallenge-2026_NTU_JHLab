import sys, anndata as ad, pandas as pd, numpy as np
from cell_eval import MetricsEvaluator
fold = sys.argv[1]
d = f"ckpts/additive/{fold}"
pred, real = ad.read_h5ad(f"{d}/pred.h5ad"), ad.read_h5ad(f"{d}/real.h5ad")
print(f"[rescore] {fold}: pred {pred.shape} real {real.shape}; perts {pred.obs['perturbation'].nunique()-1};"
      f" pred cells/pert {pred.obs['perturbation'].value_counts().drop('control').median():.0f};"
      f" real cells/pert {real.obs['perturbation'].value_counts().drop('control').median():.0f};"
      f" controls {(pred.obs['perturbation']=='control').sum()}")
ev = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert="control", pert_col="perturbation",
                      num_threads=8, outdir=f"repro/out/rescore_{fold}")
res, agg = ev.compute()
mine = agg.to_pandas().set_index("statistic").loc["mean"]
theirs = pd.read_csv(f"{d}/agg_results.csv").set_index("statistic").loc["mean"]
keys = ["pearson_delta", "mse", "mae", "discrimination_score_l1", "discrimination_score_cosine",
        "de_direction_match", "de_sig_genes_recall", "overlap_at_N", "pr_auc"]
for k in keys:
    print(f"[rescore] {k:28s} theirs {theirs[k]:.4f}  rescored {mine[k]:.4f}")
