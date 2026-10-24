"""SIGnature attributions for the fixed control-cell samples (SCimilarity encoder, Integrated Gradients).

    env/bin/python compute_attributions.py [ctx ...]

Follows the official tutorial (docs/notebooks/2-GeneratingAttributions.ipynb): SCimilarityWrapper.preprocess_adata
(align to the model's gene order, normalize_total 1e4, log1p) then calculate_attributions(method='ig',
target_sum=1e3, default 50 IG steps, zero baseline). Output per context: attr/<ctx>_mean_model.npy (mean attribution per
model gene over the 2,000 cells), attr/<ctx>_vcc.npy (the same on the 18,533-gene challenge axis, NaN where the model
has no such symbol), attr/<ctx>_stats.json.
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"
sys.path.insert(0, f"{HERE}/ext/SIGnature/src")
from SIGnature.models.scimilarity import SCimilarityWrapper

GENES = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist()
os.makedirs(f"{HERE}/attr", exist_ok=True)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
torch.manual_seed(20261013)
scim = SCimilarityWrapper(model_path=f"{HERE}/models/scimilarity", use_gpu=torch.cuda.is_available())
mgenes = list(scim.gene_order)
log(f"model: {len(mgenes)} genes in gene_order, n_genes {scim.n_genes}, latent {scim.latent_dim}, device {next(scim.model.parameters()).device}")
vcc_in_model = np.array([g in set(mgenes) for g in GENES]); midx = {g: i for i, g in enumerate(mgenes)}
log(f"challenge genes with a model symbol: {int(vcc_in_model.sum())} / {len(GENES)}")
ctxs = sys.argv[1:] or [f[:-11] for f in sorted(os.listdir(f"{HERE}/controls")) if f.endswith("_counts.npz")]
for ctx in ctxs:
    out = f"{HERE}/attr/{ctx}_vcc.npy"
    if os.path.exists(out): log(f"{ctx}: exists"); continue
    t0 = time.time()
    X = sp.load_npz(f"{HERE}/controls/{ctx}_counts.npz").tocsr(); meas = np.load(f"{HERE}/controls/{ctx}_measured.npy")
    a = ad.AnnData(X=X.copy(), obs=pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])]), var=pd.DataFrame(index=GENES))
    a.layers["counts"] = X.copy()
    ta = scim.preprocess_adata(a, gene_overlap_threshold=500)
    assert list(ta.var.index) == mgenes[:len(ta.var)] or ta.shape[1] == len(mgenes), (ta.shape, len(mgenes))
    covered = int(np.sum(meas & vcc_in_model))
    atts = scim.calculate_attributions(ta.X, batch_size=200, method="ig", disable_tqdm=True)
    atts = sp.csr_matrix(atts)
    mean_model = np.asarray(atts.mean(0)).ravel()
    frac = np.asarray((atts > 0).mean(0)).ravel()
    vcc = np.full(len(GENES), np.nan, np.float32)
    for gi, g in enumerate(GENES):
        j = midx.get(g)
        if j is not None and j < len(mean_model): vcc[gi] = mean_model[j]
    np.save(f"{HERE}/attr/{ctx}_mean_model.npy", mean_model.astype(np.float32)); np.save(out, vcc)
    # expression-side comparison on the same cells (log-normalised mean) for the record
    expr = np.asarray(ta.X.mean(0)).ravel() if sp.issparse(ta.X) else ta.X.mean(0)
    ok = (mean_model > 0) & (expr > 0)
    from scipy.stats import spearmanr
    st = {"cells": int(X.shape[0]), "seconds": round(time.time() - t0, 1), "cells_per_s": round(X.shape[0] / (time.time() - t0), 1),
          "measured_genes_in_model": covered, "genes_with_positive_mean_attribution": int((mean_model > 0).sum()),
          "attribution_row_sum_mean": float(atts.sum(1).mean()), "spearman_mean_attr_vs_mean_expr": float(spearmanr(mean_model[ok], expr[ok])[0]),
          "top10_attr_genes": [mgenes[i] for i in np.argsort(-mean_model)[:10]],
          "top10_expr_genes": [mgenes[i] for i in np.argsort(-np.asarray(expr))[:10]]}
    json.dump(st, open(f"{HERE}/attr/{ctx}_stats.json", "w"), indent=1)
    log(f"{ctx}: {st['seconds']}s ({st['cells_per_s']} cells/s), {covered} measured genes in model, top attr {st['top10_attr_genes'][:5]}")
    torch.cuda.empty_cache()
