"""Offline reach headroom from re-ordering (2026-10-20, analysis). For each hold-out target, the reference-significant
genes (own excluded) are ranked by an alternative confidence key instead of the prediction's significance; the sign
is the key's sign. reach = deepest prefix with purity >= 0.9, / N_conf. Keys (training sources only):
  pvalue      the V6 cells' actual DE ranking (as scored)
  |t| eq      |equal-weight trans + cis|        |t| d200   depth-weighted
  |t| x sqrt(mu)  scaled by the gene's control expression (power for a Wilcoxon call)
  z           |mean| / (sd / sqrt(n)) of the sources' values for that (target, gene) (+ cis kept on top)
  agree       fraction of sources agreeing with the mean's sign x |t|
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import bulk_eval as BE, sigexp as S, weights_screen as W
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
def reach_of(groups, keyfun):
    out = []
    for t, d in groups:
        rs = d[(d.p_adj_r < 0.05) & (d.feature != t) & np.isfinite(d.log2_fold_change_r) & (d.log2_fold_change_r != 0)]
        ncf = len(rs)
        if ncf == 0: continue
        key = keyfun(t, rs)
        o = np.argsort(-np.abs(key), kind="stable"); m = (np.sign(key[o]) == np.sign(rs.log2_fold_change_r.values[o])).astype(float)
        pur = np.cumsum(m) / np.arange(1, ncf + 1); ks = np.flatnonzero(pur >= 0.9)
        out.append((ks[-1] + 1) / ncf if len(ks) else 0.0)
    return float(np.mean(out))
for name in ["H1", "KOLF200"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); targets = B["targets"]; ti = {t: i for i, t in enumerate(targets)}; oc = B["oc"]
    R = pd.read_parquet(f"de_sets/{name}/de_real.parquet"); P = pd.read_parquet(f"de_sets/{name}/de_pred.parquet")
    M = R.merge(P, on=["target", "feature"], suffixes=("_r", "_p")); groups = list(M.groupby("target", sort=False))
    teq = oc + W.trans(C, train, targets, 0, W.WEIGHTS["equal"]); td = oc + W.trans(C, train, targets, 0, W.WEIGHTS["depth200"])
    # per (target, gene) source statistics
    n = np.zeros_like(oc); s1 = np.zeros_like(oc); s2 = np.zeros_like(oc); pos = np.zeros_like(oc)
    for c in train:
        Y = C[c]["Ysrc"]
        for j, t in enumerate(C[c]["targets"]):
            i = ti.get(t)
            if i is None: continue
            y = Y[j]; ok = np.isfinite(y); yz = np.where(ok, y, 0)
            n[i] += ok; s1[i] += yz; s2[i] += yz ** 2; pos[i] += (yz > 0)
    mean = s1 / np.maximum(n, 1); sd = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 1e-6))
    z = mean / (sd / np.sqrt(np.maximum(n, 1))); z = np.where(n >= 3, z, 0)
    agree = np.where(n > 0, np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1)), 0)
    cismask = (np.abs(oc) > 1e-9)
    mu = B["rate"] * 1e4
    gi = S.GI
    def mk(Mx):
        return lambda t, rs: Mx[ti[t], [gi[f] for f in rs.feature.values]]
    keys = {"pvalue": lambda t, rs: np.sign(rs.log2_fold_change_p.values) * (1e3 - np.log10(np.maximum(rs.p_value_p.values, 1e-300)) * 0 - rs.p_value_p.rank().values) ,
            "|t| eq": mk(teq), "|t| d200": mk(td), "|t| d200 x sqrt(mu)": mk(td * np.sqrt(mu)[None]),
            "z (+cis)": mk(np.where(cismask, oc * 100, z)), "agree x |t| d200": mk(td * np.where(cismask, 1.0, agree))}
    print(name, " | ".join(f"{k}: {reach_of(groups, f):.4f}" for k, f in keys.items()), flush=True)
