"""Learned non-linear combination of the sources' responses (2026-10-20, cell-free screen on the three hold-outs).

    python learned_combiner.py [H1 KOLF200 HCT116]          (pertrelay venv)

Per (target p, gene g) features from the available sources: depth-weighted mean m (ln ratio, ridge lam 3), sign agreement
a, number of sources n, across-source sd, |m_p| (the target's row norm), log expression of g in the new context's controls
and its mean over the sources. Training rows: every training context c of the hold-out acts as a pseudo-new context (its
cell-type group removed from the sources); label = c's own response y_cpg; 400 random genes per (c, p). Model: sklearn
HistGradientBoostingRegressor. Applied to the hold-out (features from all its training contexts), the predicted response
replaces the mean; same processing as G3c (shared + control axes removed, |t|^2 x (a n)^0.5 per target, matched norm).
Reported: global noise-corrected cosine, pds and the approx value vs the mean-based G3c-style trans. Hold-out truth only for
scoring.
"""
import json, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, energy_proxy_screen as EP

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
NG = 400


def stats(C, srcs, targets):
    """m (depth-weighted ridge mean, ln), a (sign agreement), n, sd per (target, gene); x_src (mean log1p ctrl)."""
    ti = {t: i for i, t in enumerate(targets)}; T = len(targets)
    num = np.zeros((T, S.G)); den = np.zeros((T, S.G)); n = np.zeros((T, S.G)); s1 = np.zeros((T, S.G)); s2 = np.zeros((T, S.G)); pos = np.zeros((T, S.G))
    xs = np.zeros(S.G); xn = 0
    for c in srcs:
        d = C[c]; Y = d["Ysrc"]; wr = d["ncell"] / (d["ncell"] + 200.0)
        xs += np.log1p(d["ctrl"].astype(np.float64)); xn += 1
        for j, t in enumerate(d["targets"]):
            i = ti.get(t)
            if i is None: continue
            y = Y[j].astype(np.float64); ok = np.isfinite(y); yz = np.where(ok, y, 0)
            num[i] += wr[j] * yz; den[i] += wr[j] * ok; n[i] += ok; s1[i] += yz; s2[i] += yz ** 2; pos[i] += yz > 0
    m = num / (den + S.LAM); mean = s1 / np.maximum(n, 1)
    a = np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))
    sd = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
    return m, a, n, sd, xs / max(xn, 1)


def features(m, a, n, sd, x_src, x_new, rows=None, cols=None):
    rn = np.linalg.norm(m, axis=1)
    if rows is None:
        return np.column_stack([m.ravel(), a.ravel(), n.ravel(), sd.ravel(), np.repeat(rn, m.shape[1]),
                                np.tile(x_new, m.shape[0]), np.tile(x_src, m.shape[0])])
    return np.column_stack([m[rows, cols], a[rows, cols], n[rows, cols], sd[rows, cols], rn[rows], x_new[cols], x_src[cols]])


def main():
    tot = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        t0 = time.time()
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        Xs, ys = [], []; rng = np.random.default_rng(0)
        for c in train:
            tl = [str(t) for t in C[c]["targets"]]
            if len(tl) < 50: continue
            srcs = [d for d in train if S.GROUP_OF[d] != S.GROUP_OF[c]]
            m, a, n, sd, xs = stats(C, srcs, tl); x_new = np.log1p(C[c]["ctrl"].astype(np.float64)); Y = C[c]["Ysrc"].astype(np.float64)
            for i in range(len(tl)):
                ok = np.flatnonzero(np.isfinite(Y[i]) & (n[i] > 0))
                if len(ok) < NG: continue
                cols = rng.choice(ok, NG, replace=False); rows = np.full(NG, i)
                Xs.append(features(m, a, n, sd, xs, x_new, rows, cols)); ys.append(Y[i, cols])
        X = np.vstack(Xs); y = np.concatenate(ys)
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200, random_state=0).fit(X, y)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        m, a, n, sd, xs = stats(C, train, targets); x_new = np.log1p(B["rate"] * 5e4)
        pred = model.predict(features(m, a, n, sd, xs, x_new)).reshape(m.shape)
        pred = np.where(n > 0, pred, 0.0)
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        ta, tn = EP.agreement(C, train, targets); f = np.sqrt((np.maximum(ta, 0) + 1e-3) * tn / tn.mean())
        def sc(t):
            t5 = E.remove_shared(t.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
            t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2 * f)[:, None]
            return BE.score(B, oc + 0.6 * t8 * (bn / np.linalg.norm(t8)))
        ref = sc(m); r = sc(pred); mix = sc(0.5 * m / np.linalg.norm(m) + 0.5 * pred / np.linalg.norm(pred))
        v = 1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']); vm = 1.94 * (mix['pds'] - ref['pds']) + 0.95 * (mix['cos'] - ref['cos'])
        tot["model"] = tot.get("model", 0) + v; tot["mix"] = tot.get("mix", 0) + vm
        print(f"{name}: {len(y):,} training rows; mean-based cos {ref['cos']:+.4f} pds {ref['pds']:.4f} | learned cos {r['cos']:+.4f} pds {r['pds']:.4f} val {v:+.4f} | "
              f"50/50 mix cos {mix['cos']:+.4f} pds {mix['pds']:.4f} val {vm:+.4f} ({time.time() - t0:.0f}s)", flush=True)
    print("sum:", tot, flush=True)


if __name__ == "__main__":
    main()
