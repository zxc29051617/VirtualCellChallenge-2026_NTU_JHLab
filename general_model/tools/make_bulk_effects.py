"""Per-context gene-total effects for render mode "dual" (V6 / V8 / V9 family), from the model's full effects and its
own + cis part.

    python make_bulk_effects.py --full FULL.npz --oc OC.npz --data DATA_DIR --contexts D E F \
        [--shared-k 10] [--ctrl-axes 50] [--scale 1.0] --out PREFIX        -> PREFIX_<ctx>.npz

trans = full - oc; remove the panel mean and the top `shared-k` principal components of trans across the panel's targets
(V6); if `ctrl-axes` > 0, also remove the top principal axes of THAT context's own control cells (log1p CP10k, 2,000 cells,
seed 0) (V8); totals = scale x (oc + trans') (V9: scale 0.5). --cells-out: cell-level effects with the shared part
doubled (V10; final candidates F8 = V10 cells + V8 totals, F9 = V10 cells + V9 totals). Uses only the predictions and the
released control cells.
G1 (2026-10-20): --gamma g sharpens per-target magnitudes (row p x |t_p|^(g-1), after the removals); --match-claim sets
totals = oc + s_c x trans' with s_c per context such that the scorer's #348 budget (across-target centred sum of squares
of the predicted pseudobulk means, bulk_lognorm 5e4, own target excluded) equals the expected claim (n_targets x the
jackknife trace of a 400-cell pseudobulk of that context's controls): below that point the correction refunds the
spread's cost, above it the plug-in error grows (repro/next_20261019/run.log). --trans-scale x: totals = oc + x trans'.
"""
import argparse, os, sys
import numpy as np, scipy.sparse as sp, anndata as ad

p = argparse.ArgumentParser()
p.add_argument("--full", required=True); p.add_argument("--oc", required=True); p.add_argument("--data", required=True)
p.add_argument("--contexts", nargs="+", required=True); p.add_argument("--shared-k", type=int, default=10)
p.add_argument("--ctrl-axes", type=int, default=0); p.add_argument("--scale", type=float, default=1.0); p.add_argument("--out", required=True)
p.add_argument("--cells-out", default=None, help="also write cell-level effects with the shared trans part doubled (V10)")
p.add_argument("--gamma", type=float, default=1.0); p.add_argument("--match-claim", action="store_true")
p.add_argument("--trans-scale", type=float, default=None)
p.add_argument("--target-weights", default=None, help="npz (targets, w): multiply each target's trans row by w after --gamma")
p.add_argument("--cis-scale", type=float, default=1.0, help="totals: own + cis-scale x cis + trans (oc minus the own gene)")
p.add_argument("--common", default=None, help="npz with a predicted common response per context (key = context, log2 on the gene axis), "
               "added to every target's totals after the removals and the budget scaling (the budget is centred across targets)")
p.add_argument("--common-scale", type=float, default=1.0)
p.add_argument("--entry-agree", type=float, default=0.0,
               help="multiply the trans part by (per-gene source sign agreement / median)^K before the removals (signature cache)")
a = p.parse_args()


def remove_shared(t, k):
    t = t - t.mean(0, keepdims=True)
    if k <= 0: return t
    _, _, vt = np.linalg.svd(t, full_matrices=False); return t - (t @ vt[:k].T) @ vt[:k]


def control_axes(C, lib, m, n=2000, seed=0):
    idx = np.random.default_rng(seed).choice(C.shape[0], min(n, C.shape[0]), replace=False)
    X = C[idx].multiply(1e4 / lib[idx][:, None]).tocsr(); X.data = np.log1p(X.data); X = X.toarray(); X -= X.mean(0, keepdims=True)
    return np.linalg.svd(X, full_matrices=False)[2][:m]


def jk_trace(X, rng, n=400, reps=5):
    """Mean jackknife trace of an n-cell pseudobulk (log1p(5e4 x fraction)) over random draws of the cells X."""
    out = []
    for _ in range(reps):
        Y = X[rng.choice(X.shape[0], n, replace=False)]
        s = np.asarray(Y.sum(0)).ravel(); T = s.sum(); f = s / T; L = np.asarray(Y.sum(1)).ravel()
        x2 = np.asarray(Y.multiply(Y).sum(0)).ravel(); xl = np.asarray(Y.T @ L).ravel()
        varf = np.maximum(x2 - 2 * f * xl + f ** 2 * (L ** 2).sum(), 0) / T ** 2
        out.append(((5e4 ** 2) * varf / (1 + 5e4 * f) ** 2).sum())
    return float(np.mean(out))


def budget(lfc, ref_mean, own):
    m = ref_mean[None] * 2.0 ** np.clip(lfc, -4, 4); mu = np.log1p(m * (5e4 / m.sum(1, keepdims=True)))
    dev = mu - mu.mean(0, keepdims=True); tot = (dev ** 2).sum()
    for i, k in enumerate(own):
        if k >= 0: tot -= dev[i, k] ** 2
    return tot


PER_CTX = "{ctx}" in a.full                         # --full path_{ctx}.npz: one input per context (trained model)
zo = np.load(a.oc, allow_pickle=True)
TW = None
if a.target_weights:
    zt = np.load(a.target_weights, allow_pickle=True); TW = dict(zip([str(t) for t in zt["targets"]], zt["w"].astype(np.float64)))


def load_trans(path):
    zf = np.load(path, allow_pickle=True)
    targets = [str(t) for t in zf["targets"]]; assert targets == [str(t) for t in zo["targets"]], "target order differs"
    full = zf["lfc"].astype(np.float64); oc = zo["lfc"].astype(np.float64)
    trans = full - oc
    if a.entry_agree:
        sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"); import sigexp as S
        ti = {t: i for i, t in enumerate(targets)}; n = np.zeros_like(full); s1 = np.zeros_like(full); pos = np.zeros_like(full)
        use = [c for c in S.CONTEXTS if c not in set(os.environ.get("VCC_EXCLUDE", "").split(","))]   # VCC_EXCLUDE: mock exams
        Cc = S.load_cache(use)
        for c in use:
            Y = Cc[c]["Ysrc"]
            for j, t in enumerate(Cc[c]["targets"]):
                i = ti.get(t)
                if i is None: continue
                y = Y[j]; ok = np.isfinite(y); yz = np.where(ok, y, 0); n[i] += ok; s1[i] += yz; pos[i] += yz > 0
        mean = s1 / np.maximum(n, 1); agree = np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))
        g = np.clip(agree / np.median(agree[np.abs(trans) > 0]), 0, None) ** a.entry_agree; g[n.sum(1) == 0] = 1.0
        trans = trans * g; del Cc
        print(f"entry agreement ^{a.entry_agree} applied to the trans part", flush=True)
    return targets, full, oc, trans


targets, full, oc, trans = load_trans(a.full.replace("{ctx}", a.contexts[0]) if PER_CTX else a.full)
t5 = remove_shared(trans, a.shared_k)
if a.cells_out:                                      # V10: cells = oc + trans + shared part (panel mean + top-k components)
    tr = full - oc; cells = oc + tr + (tr - tr.mean(0, keepdims=True) - t5)
    np.savez_compressed(a.cells_out, targets=np.array(targets), lfc=cells.astype(np.float32))
    print(f"wrote {a.cells_out} (cells: shared trans part doubled)", flush=True)
genes = None
if a.cis_scale != 1.0:                               # scale the cis neighbours (oc without each target's own gene)
    import pandas as pd
    gl = pd.read_csv(f"{a.data}/gene_names.csv").iloc[:, 0].astype(str).tolist()     # header "gene_name"
    gpos = {g: i for i, g in enumerate(gl)}; own_m = np.zeros_like(oc)
    for i, x in enumerate(targets):
        k = gpos.get(x)
        if k is not None: own_m[i, k] = oc[i, k]
    oc = own_m + a.cis_scale * (oc - own_m)
    print(f"cis neighbours x{a.cis_scale} in the totals", flush=True)
for c in a.contexts:
    if PER_CTX:
        t5 = remove_shared(load_trans(a.full.replace("{ctx}", c))[3], a.shared_k)
    t = t5
    A = ad.read_h5ad(f"{a.data}/context_{c}.h5ad"); C = sp.csr_matrix(A.X)
    if a.ctrl_axes > 0:
        lib = np.asarray(C.sum(1)).ravel()
        V = control_axes(C, lib, a.ctrl_axes); t = t5 - (t5 @ V.T) @ V
    if a.gamma != 1.0: t = t * (np.maximum(np.linalg.norm(t, axis=1), 1e-9) ** (a.gamma - 1))[:, None]
    if TW is not None: t = t * np.array([TW.get(x, np.median(list(TW.values()))) for x in targets])[:, None]
    out = f"{a.out}_{c}.npz"
    if a.match_claim or a.trans_scale is not None:
        if a.match_claim:
            if genes is None:
                genes = [str(g) for g in A.var_names]; gi = {g: i for i, g in enumerate(genes)}; own = np.array([gi.get(x, -1) for x in targets])
            Xd = C.astype(np.float64); ref_mean = np.asarray(Xd.sum(0)).ravel() / Xd.shape[0]
            claim = len(targets) * jk_trace(Xd, np.random.default_rng(0))
            lo, hi = 0.0, 1.0
            while budget(oc + hi * t, ref_mean, own) < claim: hi *= 2
            for _ in range(40):
                mid = (lo + hi) / 2; lo, hi = (mid, hi) if budget(oc + mid * t, ref_mean, own) < claim else (lo, mid)
            sc = hi; note = f"claim {claim:.0f}, s_c {sc:.4g}"
        else:
            sc = a.trans_scale; note = f"trans scale {sc}"
        lfc = oc + sc * t
    else:
        lfc = a.scale * (oc + t); note = f"scale {a.scale}"
    if a.common:                                     # predicted common response of the context (same for every target)
        cv = np.load(a.common)[c].astype(np.float64); lfc = lfc + a.common_scale * cv[None]; note += f", common x{a.common_scale}"
    np.savez_compressed(out, targets=np.array(targets), lfc=lfc.astype(np.float32))
    print(f"wrote {out} (shared_k {a.shared_k}, ctrl_axes {a.ctrl_axes}, gamma {a.gamma}, {note})", flush=True)
