"""Learn small, regularised source-weight models (groups B, C, D) with inner leave-one-group-out validation.

    python weights.py [outer ...]         (pertrelay venv; torch on CPU)

Weight of source context c for a unit u:  w_uc = N_u * softmax_c(F_uc . theta), F_uc = z-scored (within unit, across
eligible sources) features, N_u = number of eligible sources, so theta = 0 gives the uniform weights of group A.
  B: expression features only            [r_expr, r_expr_centered, coverage, source quality]
  C: B + SIGnature attribution features  [spearman, centered pearson of log attribution, top-300 jaccard]
  D: C with each context's attribution vector permuted independently across genes (own seed per context)
Contexts without single-cell controls (CD4 x3, FengNF, FengF) have no attribution: their attribution features are
missing -> 0 (neutral) after z-scoring, in C and D alike.
Objective (inner units only; the outer held-out context never enters): mean over inner units of
-mean_p cosine(y_up, s_up(w)), where y is the unit's own common-mode-removed log-ratio of target p and s the weighted
trans signature of the other contexts (vcc/gm.py estimator, lam=3), on the 4,000 genes with the largest pooled variance,
plus kappa*|theta|^2. kappa in {1e-4, 1e-3, 1e-2} chosen by inner leave-one-group-out CV (a fold removes the unit's
whole exclusion group from units AND sources). Outer evaluation never touches this.
"""
import json, os, sys, time
import numpy as np, torch
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sigexp as S

torch.set_num_threads(4); torch.manual_seed(20261013)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.float32
KAPPAS = [1e-4, 1e-3, 1e-2]; NGENES = 4000; MIN_T = 20
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
FEATS = {"B": ["r_expr", "r_expr_centered", "coverage", "quality"],
         "C": ["r_expr", "r_expr_centered", "coverage", "quality", "attr_spearman", "attr_centered", "attr_top300_jaccard"]}
FEATS["D"] = FEATS["C"]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def load_attr(shuffle):
    A = {}
    for c in S.SC_CONTEXTS:
        v = np.load(f"{HERE}/attr/{c}_vcc.npy").astype(np.float64)
        if shuffle:
            rng = np.random.default_rng([20261013, S.SC_CONTEXTS.index(c), 77]); idx = np.flatnonzero(np.isfinite(v))
            v = v.copy(); v[idx] = v[rng.permutation(idx)]
        A[c] = v
    return A


def profiles(C, outer):
    """control log profile + measured mask per context (held-out context: its 4000 NTC cells)."""
    P = {c: (C[c]["ctrl"].astype(np.float64), C[c]["measured"].astype(bool)) for c in S.CONTEXTS}
    for h in outer["held"]:
        z = np.load(f"{S.CACHE}/heldout_ctrl_{h}.npz"); P[h] = (z["ctrl"].astype(np.float64), z["measured"].astype(bool))
    return P


def features(u, sources, P, A, mu, amean, quality, use_attr):
    """raw feature matrix [len(sources), 4 or 7] for unit u (NaN = missing attribution)."""
    lu, mu_u = P[u]; xu = np.expm1(lu); expr_u = mu_u & (xu >= S.X_MIN)
    F = np.full((len(sources), 7), np.nan)
    for i, c in enumerate(sources):
        lc, mc = P[c]; xc = np.expm1(lc)
        com = mu_u & mc & (expr_u | (xc >= S.X_MIN))
        F[i, 0] = pearson(lu[com], lc[com]); F[i, 1] = pearson(lu[com] - mu[com], lc[com] - mu[com])
        F[i, 2] = (mc & expr_u).sum() / max(expr_u.sum(), 1); F[i, 3] = quality[c]
        if use_attr and u in A and c in A:
            au, ac = A[u], A[c]
            va = np.isfinite(au) & (au > 0) & mu_u; vc = np.isfinite(ac) & (ac > 0) & mc; ca = va & vc
            if ca.sum() > 500:
                F[i, 4] = pearson(rankdata(au[ca]), rankdata(ac[ca]))
                F[i, 5] = pearson(np.log(au[ca]) - amean[ca], np.log(ac[ca]) - amean[ca])
                both = mu_u & mc
                ta = set(np.argsort(-np.where(both & np.isfinite(au), au, -1))[:300]); tc = set(np.argsort(-np.where(both & np.isfinite(ac), ac, -1))[:300])
                F[i, 6] = len(ta & tc) / len(ta | tc)
    return F


def zscore(F):
    Z = F.copy()
    for j in range(F.shape[1]):
        v = F[:, j]; ok = np.isfinite(v)
        if ok.sum() >= 2 and v[ok].std() > 1e-9: Z[ok, j] = (v[ok] - v[ok].mean()) / v[ok].std()
        else: Z[ok, j] = 0.0
    return np.nan_to_num(Z, nan=0.0)                     # missing -> neutral


def softmax_weights(Z, theta):
    z = Z @ theta; e = np.exp(z - z.max()); return len(z) * e / e.sum()


def run_outer(name, outer, C, genome_unused=None):
    t0 = time.time()
    train = [c for c in S.CONTEXTS if c not in outer["exclude"]]; N = len(train)
    P = profiles(C, outer)
    mu = np.zeros(S.G); cnt = np.zeros(S.G)
    for c in train: l, m = P[c]; mu += np.where(m, l, 0); cnt += m
    mu = mu / np.maximum(cnt, 1)
    quality = {c: float(C[c]["quality"]) for c in S.CONTEXTS}
    T = S.target_list(); Ti = {t: i for i, t in enumerate(T)}
    # gene subset: largest pooled variance of Ysrc over training contexts/targets
    ss = np.zeros(S.G); n = np.zeros(S.G)
    for c in train:
        Y = C[c]["Ysrc"]; ok = np.isfinite(Y); ss += np.where(ok, Y, 0).__pow__(2).sum(0); n += ok.sum(0)
    genes_sel = np.argsort(-(ss / np.maximum(n, 1)))[:NGENES]
    Ysrc = np.zeros((len(train), len(T), NGENES), np.float32); M = np.zeros_like(Ysrc); raw = np.zeros_like(Ysrc)
    avail = np.zeros((len(train), len(T)), bool); big = np.zeros((len(train), len(T)), bool); ks = np.ones((len(train), len(T)), np.float32)
    for ci, c in enumerate(train):
        for t, i in C[c]["tidx"].items():
            j = Ti[t]; y = C[c]["Ysrc"][i][genes_sel]; ok = np.isfinite(y)
            Ysrc[ci, j] = np.where(ok, y, 0); M[ci, j] = ok; avail[ci, j] = True; big[ci, j] = C[c]["ncell"][i] >= 50
            ks[ci, j] = C[c]["kscale"][i]; raw[ci, j] = np.where(ok, y / C[c]["kscale"][i], 0)
    Yt, Mt, Rt = torch.tensor(Ysrc, device=DEV), torch.tensor(M, device=DEV), torch.tensor(raw, device=DEV)
    availt = torch.tensor(avail, device=DEV); bigt = torch.tensor(big, device=DEV)
    log(f"[{name}] train contexts {N}: {train}; genes {NGENES}; targets {len(T)}")
    res = {"outer": name, "train": train, "excluded": outer["exclude"], "held_out": outer["held"], "n_genes_inner": NGENES}
    for group in ["B", "C", "D"]:
        use_attr = group in ("C", "D"); A = load_attr(group == "D") if use_attr else {}
        amean = np.zeros(S.G)
        if use_attr:                                     # gene-wise mean of log attribution over training SC contexts
            L = np.stack([np.where(np.isfinite(A[c]) & (A[c] > 0), np.log(np.where(A[c] > 0, A[c], 1)), np.nan) for c in train if c in A])
            amean = np.nan_to_num(np.nanmean(L, 0))
        nf = len(FEATS[group])
        # per inner unit: eligible sources, standardised features
        units = {}
        for ui, u in enumerate(train):
            src = [c for c in train if S.GROUP_OF[c] != S.GROUP_OF[u]]
            Z = zscore(features(u, src, P, A, mu, amean, quality, use_attr))[:, :nf]
            units[u] = (src, Z)
        Zh = zscore(features(outer["held"][0], train, P, A, mu, amean, quality, use_attr))[:, :nf]

        def unit_loss(u, theta, avail_ctx, src_pool):
            src, Z = units[u]; keep = [i for i, c in enumerate(src) if c in src_pool]
            if len(keep) < 3: return None
            Zk = torch.tensor(Z[keep], dtype=DT, device=DEV); idx = [train.index(src[i]) for i in keep]
            z = Zk @ theta; w = len(keep) * torch.softmax(z, 0)
            Ys, Ms = Yt[idx], Mt[idx]
            s = torch.einsum("c,cpg->pg", w, Ys) / (torch.einsum("c,cpg->pg", w, Ms) + S.LAM)
            ui = train.index(u); yv = Rt[ui]; my = Mt[ui]
            nsrc = availt[idx].sum(0); valid = availt[ui] & bigt[ui] & (nsrc >= 3)
            if int(valid.sum()) < MIN_T: return None
            num = (yv * s * my).sum(1); d = torch.sqrt(((yv * my) ** 2).sum(1) * ((s * my) ** 2).sum(1) + 1e-12)
            return -(num / d)[valid].mean()

        def total_loss(theta, fit_units, src_pool, kappa):
            ls = [unit_loss(u, theta, None, src_pool) for u in fit_units]; ls = [l for l in ls if l is not None]
            return (torch.stack(ls).mean() if ls else torch.zeros((), dtype=DT, device=DEV)) + kappa * (theta ** 2).sum(), len(ls)

        def fit(fit_units, src_pool, kappa):
            th = torch.zeros(nf, dtype=DT, device=DEV, requires_grad=True)
            opt = torch.optim.LBFGS([th], lr=1.0, max_iter=40, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad(); l, _ = total_loss(th, fit_units, src_pool, kappa); l.backward(); return l
            opt.step(closure); return th.detach()

        # inner leave-one-group-out CV for each kappa
        cv = {}
        base_cv = None
        for kappa in KAPPAS:
            fold_losses = []; base_losses = []
            for gi_, grp in enumerate(S.GROUPS):
                held = [c for c in grp if c in train]
                if not held: continue
                pool = set(c for c in train if c not in held)
                fit_units = [u for u in train if u not in held]
                th = fit(fit_units, pool, kappa)
                ev = [unit_loss(u, th, None, pool) for u in held]; b0 = [unit_loss(u, torch.zeros(nf, dtype=DT, device=DEV), None, pool) for u in held]
                ev = [float(x) for x in ev if x is not None]; b0 = [float(x) for x in b0 if x is not None]
                if ev: fold_losses.append(np.mean(ev)); base_losses.append(np.mean(b0))
            cv[kappa] = {"cv_loss": float(np.mean(fold_losses)), "uniform_loss": float(np.mean(base_losses)), "n_folds": len(fold_losses),
                         "gain_over_uniform": float(np.mean(base_losses) - np.mean(fold_losses))}
            log(f"[{name}/{group}] kappa {kappa}: inner CV loss {cv[kappa]['cv_loss']:.5f} vs uniform {cv[kappa]['uniform_loss']:.5f}")
        best = min(KAPPAS, key=lambda k: cv[k]["cv_loss"])
        pool_all = set(train); th = fit(train, pool_all, best)
        th = th.cpu(); w = softmax_weights(Zh, th.numpy().astype(np.float64))
        res[group] = {"features": FEATS[group], "kappa": best, "theta": th.numpy().tolist(), "inner_cv": {str(k): v for k, v in cv.items()},
                      "weights": {c: float(x) for c, x in zip(train, w)}, "features_heldout_z": {c: Zh[i].tolist() for i, c in enumerate(train)}}
        log(f"[{name}/{group}] chosen kappa {best}, theta {np.round(th.numpy(), 3).tolist()}; weights " +
            ", ".join(f"{c}:{x:.2f}" for c, x in sorted(zip(train, w), key=lambda t: -t[1])[:5]))
    res["A"] = {"weights": {c: 1.0 for c in train}}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{HERE}/weights_{name}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    C = S.load_cache()
    for name in sys.argv[1:] or list(S.OUTER):
        run_outer(name, S.OUTER[name], C)
