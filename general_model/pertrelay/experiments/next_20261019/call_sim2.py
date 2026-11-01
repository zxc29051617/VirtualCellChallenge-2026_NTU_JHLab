"""Offline fid / jac / reach if the predicted-significant set were the top-n genes by a confidence key (2026-10-20).
n = c x N_conf (oracle size, headroom) or n = fixed. Key = agreement x |depth-weighted trans + cis| (cis on top)."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import bulk_eval as BE, sigexp as S, weights_screen as W
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
def sim(groups, key_of, nrule):
    fids, jacs, reaches, npreds, ncfs = [], [], [], [], []
    for t, d in groups:
        d = d[(d.feature != t) & np.isfinite(d.log2_fold_change_r)]
        lr = d.log2_fold_change_r.values; real_sig = (d.p_adj_r < 0.05).values; ncf = int(real_sig.sum())
        key = key_of(t, d); o = np.argsort(-np.abs(key), kind="stable")
        n = nrule(ncf, key); call = np.zeros(len(d), bool); call[o[:n]] = True
        indenom = lr != 0; match = indenom & (np.sign(key) == np.sign(lr))
        sc = call & indenom; n_pred = int(sc.sum()); k = int((sc & match).sum())
        fids.append(k / max(n_pred, ncf) if max(n_pred, ncf) > 0 else 1.0)
        jacs.append((call & real_sig).sum() / max((call | real_sig).sum(), 1))
        rs = [i for i in o if real_sig[i] and indenom[i]]
        if ncf > 0:
            m = match[rs].astype(float); pur = np.cumsum(m) / np.arange(1, len(rs) + 1); ks = np.flatnonzero(pur >= 0.9)
            reaches.append((ks[-1] + 1) / ncf if len(ks) else 0.0)
        npreds.append(n_pred); ncfs.append(ncf)
    return np.mean(fids), np.mean(jacs), np.mean(reaches), np.median(npreds), np.median(ncfs)
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
    cis = np.abs(oc) > 1e-9
    K = np.where(cis, oc * 10, td * agree)                       # cis genes first, then agreement-weighted trans
    gi = S.GI; key_of = lambda t, d: K[ti[t], [gi[f] for f in d.feature.values]]
    pv_of = lambda t, d: np.sign(d.log2_fold_change_p.values) * (1.0 / (1e-300 + d.p_value_p.values)) ** 0.001 * (d.p_adj_p.values < 0.05 + 0)
    print(name, flush=True)
    for lab, kf in [("cells p-value order", pv_of), ("agree x |t| key", key_of)]:
        for c in [0.5, 1.0, 1.5, 2.0]:
            f, j, r, npd, ncf = sim(groups, kf, lambda ncf, key, c=c: int(round(c * ncf)))
            print(f"   {lab:20s} n = {c} x N_conf: fid {f:.4f} jac {j:.4f} reach {r:.4f} (median n_pred {npd:.0f}, N_conf {ncf:.0f})", flush=True)
        for nf in [50, 150, 400]:
            f, j, r, npd, ncf = sim(groups, kf, lambda ncf, key, nf=nf: nf)
            print(f"   {lab:20s} n = {nf:<4d}         fid {f:.4f} jac {j:.4f} reach {r:.4f}", flush=True)


# --- threshold rules: call genes whose key exceeds a global threshold (n_pred adapts to each target's strength)
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "thr":
    pass
