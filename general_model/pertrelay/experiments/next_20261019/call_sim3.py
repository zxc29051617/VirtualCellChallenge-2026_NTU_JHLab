"""Threshold calling rule (2026-10-20, offline): call genes with |key| above a global quantile tau of all keys (per
hold-out), so strong responders get more calls. Reports fid / jac / reach and the median and spread of n_pred vs N_conf,
and the correlation of n_pred with N_conf across targets."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, ".")
import bulk_eval as BE, sigexp as S, weights_screen as W, call_sim2 as CS2
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
for name in ["H1", "KOLF200"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); targets = B["targets"]; ti = {t: i for i, t in enumerate(targets)}; oc = B["oc"]
    R = pd.read_parquet(f"de_sets/{name}/de_real.parquet"); P = pd.read_parquet(f"de_sets/{name}/de_pred.parquet")
    M = R.merge(P, on=["target", "feature"], suffixes=("_r", "_p")); groups = list(M.groupby("target", sort=False))
    td = W.trans(C, train, targets, 0, W.WEIGHTS["depth200"])
    n = np.zeros_like(oc); s1 = np.zeros_like(oc); pos = np.zeros_like(oc)
    for c in train:
        Y = C[c]["Ysrc"]
        for j, t in enumerate(C[c]["targets"]):
            i = ti.get(t)
            if i is None: continue
            y = Y[j]; ok = np.isfinite(y); n[i] += ok; s1[i] += np.where(ok, y, 0); pos[i] += (np.where(ok, y, 0) > 0)
    mean = s1 / np.maximum(n, 1); agree = np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))
    cis = np.abs(oc) > 1e-9; mu = B["rate"] * 1e4
    for kname, Kt in [("agree x |t|", td * agree), ("agree x |t| x mu^0.25", td * agree * mu[None] ** 0.25)]:
        K = np.where(cis, oc * 1e3, Kt); gi = S.GI
        key_of = lambda t, d: K[ti[t], [gi[f] for f in d.feature.values]]
        allk = np.abs(Kt[:, mu > 0.05]).ravel()
        for q in [0.99, 0.995, 0.998, 0.999]:
            tau = np.quantile(allk, q)
            res = {}
            def rule(ncf, key, tau=tau):
                nn = int((np.abs(key) > tau).sum()); res.setdefault("pairs", []).append((nn, ncf)); return nn
            f, j, r, npd, ncf = CS2.sim(groups, key_of, rule)
            pr = np.array(res["pairs"]); rho = spearmanr(pr[:, 0], pr[:, 1]).correlation
            print(f"{name} {kname:22s} tau q{q}: fid {f:.4f} jac {j:.4f} reach {r:.4f} | n_pred median {np.median(pr[:, 0]):.0f} (IQR {np.percentile(pr[:, 0], 25):.0f}-{np.percentile(pr[:, 0], 75):.0f}), N_conf median {ncf:.0f}, spearman(n_pred, N_conf) {rho:.2f}", flush=True)
