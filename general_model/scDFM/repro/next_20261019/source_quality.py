"""Per-source transfer quality on each hold-out (2026-10-20, analysis): global noise-corrected cosine of ONE source's
trans responses (same post-processing) with the hold-out truth, on the targets that source covers, vs the pooled
prediction on the same targets; and per-target energy predictors (Spearman with the true energy)."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, coexp_screen as CS
SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"
for name in ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; Cc = S.load_cache(train)
    B = BE.load(name); D = B["D"].copy(); oc = B["oc"]; targets = B["targets"]; ti = {t: i for i, t in enumerate(targets)}
    for i, k in enumerate(B["own"]):
        if k >= 0: D[i, k] = 0
    dist = (D ** 2).sum(1) - B["v"]
    Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    prep = lambda t: (lambda t5: t5 - (t5 @ Vc.T) @ Vc)(E.remove_shared(t).astype(np.float64))
    def cosg(T, rows):
        Q = CS.pred_deltas(oc[rows] + 0.6 * T[rows], B["rate"], B["mu_c"])
        for j, i in enumerate(rows):
            k = B["own"][i]
            if k >= 0: Q[j, k] = 0
        return (Q * D[rows]).sum() / np.sqrt((Q ** 2).sum() * dist[rows].sum())
    pooled = prep(B["full"] - oc)
    print(f"{name}: pooled cos (all targets) {cosg(pooled, np.arange(len(targets))):+.3f}", flush=True)
    rms = {}
    for c in train:
        Y = Cc[c]["Ysrc"]; rows = []; T = np.zeros_like(oc)
        for j, t in enumerate(Cc[c]["targets"]):
            i = ti.get(t)
            if i is not None: T[i] = np.nan_to_num(Y[j]) * S.ALPHA / np.log(2); rows.append(i); rms.setdefault(i, []).append(np.sqrt(np.nanmean(Y[j] ** 2)))
        if len(rows) < 30: continue
        rows = np.array(rows); Tp = prep(T); Tp[np.setdiff1d(np.arange(len(targets)), rows)] = 0
        print(f"   {c:13s} n {len(rows):3d}  cos alone {cosg(Tp, rows):+.3f}  pooled on same targets {cosg(pooled, rows):+.3f}", flush=True)
    nq = np.linalg.norm(pooled, axis=1)
    m_mean = np.array([np.mean(rms[i]) if i in rms else np.nan for i in range(len(targets))]); m_max = np.array([np.max(rms[i]) if i in rms else np.nan for i in range(len(targets))])
    ok = np.isfinite(m_mean)
    print(f"   energy predictors (spearman with true energy): |pooled trans| {spearmanr(nq, dist).correlation:.3f}, source rms mean {spearmanr(m_mean[ok], dist[ok]).correlation:.3f}, max {spearmanr(m_max[ok], dist[ok]).correlation:.3f}", flush=True)
