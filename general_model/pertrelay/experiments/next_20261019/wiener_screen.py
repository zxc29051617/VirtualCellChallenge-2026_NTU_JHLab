"""Per-component transfer filter (a diagonal Wiener filter in the PCA basis of the predictions), screened on the three
hold-outs (2026-10-20).

    python wiener_screen.py [H1 KOLF200 HCT116]          (pertrelay venv)

Within each hold-out's training contexts, every source c is predicted from the others (its cell-type group left out,
same ridge as the model): rows X = s_-c[p], Y = y_c[p] (NaN -> 0). In the basis of the top-K principal directions v_k of
X, beta_k = <X v_k, Y v_k> / |X v_k|^2 says how much of a prediction's component k reappears in an unseen context;
beta_rest the same for the remainder. The hold-out trans part becomes sum_k beta_k (s.v_k) v_k + beta_rest x rest, then
the usual shared-component and control-axis removal; totals = own + cis + s x trans at matched norm. Optional per-source
PC cleaning first (pcclean k). No hold-out data in the filter.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def source_mats(C, train, kclean):
    out = {}
    for c in train:
        Y = C[c]["Ysrc"].astype(np.float64); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        if kclean > 0:
            _, _, vt = np.linalg.svd(Yz - Yz.mean(0, keepdims=True), full_matrices=False)
            Yz = np.where(ok, Yz - (Yz @ vt[:kclean].T) @ vt[:kclean], 0)
        out[c] = (list(C[c]["targets"]), Yz, ok)
    return out


def ridge(M, cs, targets):
    ti = {t: i for i, t in enumerate(targets)}; num = np.zeros((len(targets), S.G)); den = np.zeros((len(targets), S.G))
    for c in cs:
        tl, Yz, ok = M[c]
        for j, t in enumerate(tl):
            i = ti.get(t)
            if i is not None: num[i] += Yz[j]; den[i] += ok[j]
    return num / (den + S.LAM), den


def fit_filter(M, train, K):
    Xs, Ys = [], []
    for c in train:
        rest = [d for d in train if S.GROUP_OF[d] != S.GROUP_OF[c]]
        tl, Yz, ok = M[c]; X, den = ridge(M, rest, tl)
        keep = den.sum(1) > 0; X = np.where(ok, X, 0)[keep]; Xs.append(X); Ys.append(Yz[keep])
    X = np.vstack(Xs); Y = np.vstack(Ys)
    gram = X @ X.T; w, U = np.linalg.eigh(gram); o = np.argsort(-w)[:K]
    V = (U[:, o].T @ X) / np.sqrt(np.maximum(w[o], 1e-12))[:, None]           # top-K right singular vectors (K x G)
    xs, ys = X @ V.T, Y @ V.T
    beta = (xs * ys).sum(0) / np.maximum((xs ** 2).sum(0), 1e-12)
    Xr = X - xs @ V; Yr = Y - ys @ V
    beta_rest = float((Xr * Yr).sum() / max((Xr ** 2).sum(), 1e-12))
    return V, beta, beta_rest


def main():
    res = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        prep = lambda t: (lambda t5: t5 - (t5 @ Vc.T) @ Vc)(E.remove_shared(t).astype(np.float64))
        base_norm = np.linalg.norm(prep(B["full"] - oc)); row = {}
        for kclean in [0, 10]:
            M = source_mats(C, train, kclean); sfull, _ = ridge(M, train, targets)
            for K in [0, 20, 50, 100, 200]:
                if K == 0: t = sfull
                else:
                    V, beta, br = fit_filter(M, train, K); proj = sfull @ V.T
                    t = (proj * np.clip(beta, 0, None)) @ V + max(br, 0) * (sfull - proj @ V)
                    if kclean == 0 and K in (20, 200): print(f"  {name} K{K}: beta top-10 {np.round(beta[:10], 2).tolist()} ... rest {br:.3f}", flush=True)
                t8 = prep(S.ALPHA * t / np.log(2)); nrm = np.linalg.norm(t8) / base_norm
                for s in [0.1, 0.3]:
                    row[f"clean{kclean} K{K} s{s}"] = BE.score(B, oc + (s / nrm) * t8)
                print(f"  {name} clean{kclean} K{K:<4d} " + "  ".join(f"s{s}: pds {row[f'clean{kclean} K{K} s{s}']['pds']:.4f} mse {row[f'clean{kclean} K{K} s{s}']['mse_free']:.4f}" for s in [0.1, 0.3]), flush=True)
        res[name] = row
    json.dump(res, open("wiener_screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
