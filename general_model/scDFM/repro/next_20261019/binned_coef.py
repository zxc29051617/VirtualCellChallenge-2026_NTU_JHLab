"""Binned transfer coefficients (2026-10-20, cell-free): prediction = m x g(bin), g per bin of (sign agreement, number of
sources, the gene's log expression in the new context, sd/|m|) fitted as sum(y m) / sum(m^2) over pseudo-new contexts
(each training context with its cell-type group removed from the sources; all genes). A learned version of the agreement
weighting. Applied to the hold-out, then the G3c processing. cos / pds / approx value vs the mean-based G3c-style trans."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, energy_proxy_screen as EP, learned_combiner as LC
SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"
AE = [0, 0.55, 0.65, 0.75, 0.85, 1.01]; NE = [0, 3, 6, 9, 99]; XE = [-1, 0.5, 1.5, 3, 99]; RE = [0, 0.5, 1, 2, 1e9]


def bins(a, n, x, r):
    return (np.digitize(a, AE[1:-1]) * (len(NE) - 1) + np.digitize(n, NE[1:-1])) * ((len(XE) - 1) * (len(RE) - 1)) + \
        np.digitize(x, XE[1:-1]) * (len(RE) - 1) + np.digitize(r, RE[1:-1])


NB = (len(AE) - 1) * (len(NE) - 1) * (len(XE) - 1) * (len(RE) - 1)
tot = {}
for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
    t0 = time.time()
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    num = np.zeros(NB); den = np.zeros(NB)
    for c in train:
        tl = [str(t) for t in C[c]["targets"]]
        if len(tl) < 50: continue
        srcs = [d for d in train if S.GROUP_OF[d] != S.GROUP_OF[c]]
        m, a, n, sd, xs = LC.stats(C, srcs, tl); x_new = np.log1p(C[c]["ctrl"].astype(np.float64)); Y = C[c]["Ysrc"].astype(np.float64)
        ok = np.isfinite(Y) & (n > 0); r = sd / np.maximum(np.abs(m), 1e-6)
        b = bins(a, n, np.broadcast_to(x_new, m.shape), r)
        np.add.at(num, b[ok], (Y * m)[ok]); np.add.at(den, b[ok], (m * m)[ok])
    g = np.clip(num / np.maximum(den, 1e-9), 0, None); g0 = num.sum() / den.sum(); g = np.where(den > 1e-3, g, g0)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]
    m, a, n, sd, xs = LC.stats(C, train, targets); x_new = np.log1p(B["rate"] * 5e4); r = sd / np.maximum(np.abs(m), 1e-6)
    pred = m * g[bins(a, n, np.broadcast_to(x_new, m.shape), r)]
    V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64); bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
    ta, tn = EP.agreement(C, train, targets); f = np.sqrt((np.maximum(ta, 0) + 1e-3) * tn / tn.mean())
    ag = np.clip(a / np.median(a[np.abs(m) > 0]), 0, None)
    def sc(t):
        t5 = E.remove_shared(t.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
        t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2 * f)[:, None]
        return BE.score(B, oc + 0.6 * t8 * (bn / np.linalg.norm(t8)))
    ref = sc(m); r1 = sc(m * ag); r2 = sc(pred)
    v1 = 1.94 * (r1['pds'] - ref['pds']) + 0.95 * (r1['cos'] - ref['cos']); v2 = 1.94 * (r2['pds'] - ref['pds']) + 0.95 * (r2['cos'] - ref['cos'])
    tot["agree"] = tot.get("agree", 0) + v1; tot["binned"] = tot.get("binned", 0) + v2
    print(f"{name}: g range {g.min():.3f}-{g.max():.3f} (global {g0:.3f}) | mean cos {ref['cos']:+.4f} pds {ref['pds']:.4f} | x agreement val {v1:+.4f} | binned coef cos {r2['cos']:+.4f} pds {r2['pds']:.4f} val {v2:+.4f} ({time.time() - t0:.0f}s)", flush=True)
print("sum:", tot, flush=True)
