"""Which per-target magnitude rule best matches where the real response is? (2026-10-20, cell-free)

    python energy_proxy_screen.py [H1 KOLF200 HCT116]          (scDFM venv)

G1 sharpens row p of the trans part by |t_p|^2 (gamma 3, |t_p| = the processed depth-weighted trans norm). Alternatives
set row p's norm to a proxy e_p (direction unchanged): agreement a_p = mean pairwise cosine between the training sources'
responses to p (over genes both measure), the number of sources n_p, and products. Reported: global noise-corrected cosine
(mse at budget = claim), pds, and Spearman(proxy, true energy). Training sources only; hold-out truth for scoring.
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W

SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"


def agreement(C, train, targets):
    ti = {t: i for i, t in enumerate(targets)}; rows = {t: [] for t in targets}
    for c in train:
        Y = C[c]["Ysrc"]
        for j, t in enumerate(C[c]["targets"]):
            if t in rows: rows[t].append(Y[j])
    a = np.zeros(len(targets)); n = np.zeros(len(targets))
    for t, ys in rows.items():
        i = ti[t]; n[i] = len(ys); cs = []
        for u in range(len(ys)):
            for v in range(u + 1, len(ys)):
                m = np.isfinite(ys[u]) & np.isfinite(ys[v])
                if m.sum() > 500:
                    x, y = ys[u][m], ys[v][m]; cs.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
        a[i] = np.mean(cs) if cs else 0.0
    return a, n


def main():
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]; D = B["D"].copy()
        for i, k in enumerate(B["own"]):
            if k >= 0: D[i, k] = 0
        energy = (D ** 2).sum(1) - B["v"]
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        t5 = E.remove_shared(W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]).astype(np.float32)).astype(np.float64)
        t8 = t5 - (t5 @ V.T) @ V; nr = np.maximum(np.linalg.norm(t8, axis=1), 1e-9); unit = t8 / nr[:, None]
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        a, n = agreement(C, train, targets); ap = np.maximum(a, 0) + 1e-3
        proxies = {"|t|^3 (G1)": nr ** 3, "|t|^2": nr ** 2, "a": ap, "a^2": ap ** 2, "|t|^3 x a": nr ** 3 * ap,
                   "|t|^3 x a^2": nr ** 3 * ap ** 2, "|t|^2 x a": nr ** 2 * ap, "|t|^3 x n": nr ** 3 * n}
        out = [f"spearman(a, energy) {spearmanr(a, energy).correlation:.3f}, (|t|, energy) {spearmanr(nr, energy).correlation:.3f}"]
        ref = None
        for k, e in proxies.items():
            t = unit * e[:, None]; r = BE.score(B, oc + 0.6 * t * (bn / np.linalg.norm(t)))
            if ref is None: ref = r
            out.append(f"{k}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']):+.4f}")
        print(name, " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
