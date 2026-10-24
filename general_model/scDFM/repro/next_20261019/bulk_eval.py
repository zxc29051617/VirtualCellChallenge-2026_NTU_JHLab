"""Fast, cell-free evaluation of gene totals on the three hold-outs (2026-10-20), for screening bulk-level changes.

    from bulk_eval import load, score          (scDFM venv)

For a hold-out set, load(name) returns the real per-target pseudobulk deltas in the scorer's space (bulk_lognorm 5e4,
vs the set's NTC cells), the real side's sampling-noise trace per target (delta-method jackknife of the pseudobulk,
own target gene excluded) and the model components (own + cis, full trans x3, shuffled). score(B, lfc_bulk) gives
  mse_free  sum_p(|q_p - r_p|^2 - v_p) / sum_p(|r_p|^2 - v_p) over all genes except p's own target: the official mse
            member's estimator WITHOUT the prediction-side correction (on a deep panel such as A/B/C that correction is
            small and fully granted; on the shallow hold-outs #348's budget distorts the official raw number)
  pds       noise-free pds proxy (cosine, every panel target gene removed) = official raw pds for exact totals
  cos       global noise-corrected cosine sum_p q_p.r_p / sqrt(sum_p |q_p|^2 sum_p (|r_p|^2 - v_p)), own gene excluded: what
            the mse member rewards once the totals sit at #348's budget = claim point (raw mse ~ 1 - 2 cos sqrt(claim/dist))
Exact totals: the prediction's pseudobulk is rate x 2^lfc renormalised (no sampling). Uses hold-out truth for scoring.
"""
import sys
import numpy as np, scipy.sparse as sp, anndata as ad

sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM"); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/signature_20261013")
import coexp_screen as CS, render_eval as E, sigexp as S

ROOT = "/home/zxc29051617/VCC"


def load(name):
    import json
    R = ad.read_h5ad(f"{ROOT}/data/stage3/loco/{name}/real.h5ad"); X = sp.csr_matrix(R.X).astype(np.float64)
    lab = R.obs.target_gene.astype(str).values
    targets = json.load(open(f"{ROOT}/data/stage3/loco/{name}/real.json"))["targets"]
    ctrl = np.asarray(X[lab == "non-targeting"].sum(0)).ravel()
    mu_c = CS.bulk(ctrl[None])[0]; rate = ctrl / ctrl.sum()
    D = np.zeros((len(targets), S.G)); v = np.zeros(len(targets)); own = np.full(len(targets), -1)
    for i, t in enumerate(targets):
        Y = X[lab == t]; s = np.asarray(Y.sum(0)).ravel(); T = s.sum(); f = s / T; L = np.asarray(Y.sum(1)).ravel()
        D[i] = np.log1p(5e4 * f) - mu_c
        x2 = np.asarray(Y.multiply(Y).sum(0)).ravel(); xl = np.asarray(Y.T @ L).ravel()
        varf = np.maximum(x2 - 2 * f * xl + f ** 2 * (L ** 2).sum(), 0) / T ** 2          # sum_i (x_ig - f_g L_i)^2 / T^2
        vg = (5e4 ** 2) * varf / (1 + 5e4 * f) ** 2
        k = S.GI.get(t)
        if k is not None: vg[k] = 0; own[i] = k
        v[i] = vg.sum()
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True
    _, oc, full, shuf = E.components(name)
    C = X[lab == "non-targeting"].tocsr(); lib = np.asarray(C.sum(1)).ravel()
    return dict(name=name, targets=targets, D=D, v=v, own=own, rate=rate, mu_c=mu_c, mask=~panel & (rate > 0), C=C, lib=lib,
                oc=oc.astype(np.float64), full=full.astype(np.float64), shuf=shuf.astype(np.float64))


def score(B, lfc_bulk, per_target=False):
    Q = CS.pred_deltas(lfc_bulk, B["rate"], B["mu_c"]); D = B["D"]
    err = ((Q - D) ** 2).sum(1); dist = (D ** 2).sum(1)
    for i, k in enumerate(B["own"]):
        if k >= 0: err[i] -= (Q[i, k] - D[i, k]) ** 2; dist[i] -= D[i, k] ** 2
    mse = float((err - B["v"]).sum() / (dist - B["v"]).sum())
    qd = (Q * D).sum(1); qq = (Q ** 2).sum(1)
    for i, k in enumerate(B["own"]):
        if k >= 0: qd[i] -= Q[i, k] * D[i, k]; qq[i] -= Q[i, k] ** 2
    cosg = float(qd.sum() / np.sqrt(qq.sum() * (dist - B["v"]).sum()))
    out = dict(mse_free=round(mse, 4), pds=round(CS.pds_like(Q, D, B["mask"]), 4), cos=round(cosg, 4))
    if per_target: out["err"], out["dist"] = err - B["v"], dist - B["v"]
    return out
