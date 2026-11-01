"""Turn a target's predicted log2 fold changes into counts per cell.

mode "current": as vccsub/make_effect_submission.py -- for fc < 1 each count is binomially thinned with p = fc; for
                fc > 1 Poisson(lib x rate x (fc - 1)) counts are added. The pseudobulk carries Monte-Carlo noise from
                the control-subset draw and from the thinning/Poisson step.
mode "exact":   "current", then each gene's total over the target's cells is set exactly to round(n x ref_mean x fc),
                ref_mean = mean count per cell over the whole control pool: the few residual counts are added in
                proportion to library size or removed uniformly from the existing counts. The pseudobulk equals the
                prediction; the per-cell noise structure is kept.
mode "dual":    cells generated with lfc_cells (cell-level amplitude), gene totals set exactly to
                round(n x ref_mean x 2^lfc_bulk) (bulk amplitude). Counts needed to reach the bulk total are added to,
                or removed from, the upper tail of the target's cells (largest current counts), so most cells keep the
                cell-level change while the pseudobulk follows the bulk amplitude.
mode "dual_lib": as "dual", but the adjustment is made in the cells with the largest libraries (additions in the top
                10 %, removals capped within the top 25 %), where a count moves the per-cell normalised value least.
mode "mix":     as "dual", but the cell-level change of a gene with |lfc| < 1/kappa is carried by a random fraction
                pi = kappa |lfc| of the cells, each changed by the fixed factor 2^(+-1/kappa) (the other cells unchanged);
                |lfc| >= 1/kappa: all cells by 2^lfc as before. The geometric-mean change (pdex log2FC) stays ~lfc, but
                the Wilcoxon statistic then grows with |lfc| itself rather than with |lfc| x sqrt(expression), so the
                prediction's significance order follows the predicted effect size (2026-10-20; reach ranks the reference's
                significant genes by OUR p-values, and fid counts our significant calls).
env DOWN_HET=1 (2026-10-23, modes other than "mix"): down-regulated genes vanish from a share 1 - fc of the cells instead of
                being thinned in every cell (same expected total; real knockdown responses of down genes are heterogeneous).
"""
import os
import numpy as np
import scipy.sparse as sp
from numba import njit

LFC_CLIP = 4.0
TOP_FRAC = 0.1


@njit(cache=True)
def _fix_exact(X, T, lib_cum, seed):
    np.random.seed(seed)
    n, G = X.shape
    for g in range(G):
        S = 0
        for c in range(n): S += X[c, g]
        r = T[g] - S
        while r > 0:                                    # add one count to a cell drawn in proportion to library size
            u = np.random.random() * lib_cum[n - 1]
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lib_cum[mid] > u: hi = mid
                else: lo = mid + 1
            X[lo, g] += 1; r -= 1
        while r < 0 and S > 0:                          # remove one existing count, uniformly over counts
            u = np.random.randint(0, S); acc = 0
            for c in range(n):
                acc += X[c, g]
                if acc > u:
                    X[c, g] -= 1; break
            S -= 1; r += 1


@njit(cache=True)
def _fix_dual(X, T, top_frac, seed):
    np.random.seed(seed)
    n, G = X.shape
    m = max(1, int(np.ceil(top_frac * n)))
    jit = np.empty(n)
    for g in range(G):
        S = 0
        for c in range(n): S += X[c, g]
        r = T[g] - S
        if r == 0: continue
        for c in range(n): jit[c] = X[c, g] + 0.5 * np.random.random()   # random tie-break
        order = np.argsort(-jit)
        if r > 0:                                       # add to the top-m cells in proportion to (count + 1)
            wsum = 0.0
            for i in range(m): wsum += X[order[i], g] + 1.0
            given = 0
            frac = np.empty(m)
            for i in range(m):
                share = r * (X[order[i], g] + 1.0) / wsum
                k = int(np.floor(share)); X[order[i], g] += k; given += k; frac[i] = share - k
            left = r - given
            fo = np.argsort(-(frac + 1e-9 * np.random.random(m)))
            for i in range(left): X[order[fo[i]], g] += 1
        else:                                           # remove by lowering the top cells to a common level
            k = -r
            if k >= S:
                for c in range(n): X[c, g] = 0
                continue
            vals = np.empty(n, np.int64)
            for i in range(n): vals[i] = X[order[i], g]
            j = 1; prefix = vals[0]                     # prefix = sum of the j largest counts
            while j < n:
                if prefix - j * vals[j] >= k: break     # lowering the top j cells to vals[j] frees enough
                prefix += vals[j]; j += 1
            keep = prefix - k                           # what the top-j cells hold after removal
            base = keep // j; extra = keep - base * j
            for i in range(j):
                X[order[i], g] = base + (1 if i < extra else 0)


@njit(cache=True)
def _cap_subset(X, g, cells, k):
    """Remove k counts of gene g from the given cells by lowering their largest values to a common level."""
    m = len(cells)
    vals = np.empty(m, np.int64)
    for i in range(m): vals[i] = X[cells[i], g]
    order = np.argsort(-(vals + 0.5 * np.random.random(m)))
    sv = np.empty(m, np.int64)
    for i in range(m): sv[i] = vals[order[i]]
    total = 0
    for i in range(m): total += sv[i]
    if k >= total:
        for i in range(m): X[cells[i], g] = 0
        return total
    j = 1; prefix = sv[0]
    while j < m:
        if prefix - j * sv[j] >= k: break
        prefix += sv[j]; j += 1
    keep = prefix - k; base = keep // j; extra = keep - base * j
    for i in range(j): X[cells[order[i]], g] = base + (1 if i < extra else 0)
    return k


@njit(cache=True)
def _fix_dual_lib(X, T, lib, add_frac, rem_frac, seed):
    """As _fix_dual, but the adjustment goes to the cells with the LARGEST libraries, where one count moves the
    per-cell normalised value least: additions into the top add_frac of cells by library (in proportion to count + 1),
    removals capped within the top rem_frac of cells by library (then all cells if that is not enough)."""
    np.random.seed(seed)
    n, G = X.shape
    by_lib = np.argsort(-(lib + 1e-3 * np.random.random(n)))
    ma = max(1, int(np.ceil(add_frac * n))); mr = max(1, int(np.ceil(rem_frac * n)))
    top_add = by_lib[:ma]; top_rem = by_lib[:mr]; allc = by_lib.copy()
    frac = np.empty(ma)
    for g in range(G):
        S = 0
        for c in range(n): S += X[c, g]
        r = T[g] - S
        if r > 0:
            wsum = 0.0
            for i in range(ma): wsum += X[top_add[i], g] + 1.0
            given = 0
            for i in range(ma):
                share = r * (X[top_add[i], g] + 1.0) / wsum
                k = int(np.floor(share)); X[top_add[i], g] += k; given += k; frac[i] = share - k
            fo = np.argsort(-(frac + 1e-9 * np.random.random(ma)))
            for i in range(r - given): X[top_add[fo[i]], g] += 1
        elif r < 0:
            k = -r
            got = _cap_subset(X, g, top_rem, k)
            if got < k: _cap_subset(X, g, allc, k - got)


KAPPA = 0.5


def render(C_draw, lib, rate, ref_mean, lfc_cells, rng, mode="current", lfc_bulk=None, fix_seed=0):
    """C_draw: counts of the drawn control cells (n x G, sparse or dense); lib: their library sizes; rate: control
    count per unit library per gene; ref_mean: mean count per cell over the whole control pool. Returns CSR float32."""
    X0 = C_draw.toarray() if sp.issparse(C_draw) else np.asarray(C_draw)
    X0 = np.rint(X0).astype(np.int64)
    fc = 2.0 ** np.clip(np.nan_to_num(lfc_cells), -LFC_CLIP, LFC_CLIP)
    X = X0.copy()
    if mode == "mix":                                   # fraction pi = kappa |lfc| of the cells changed by 2^(+-1/kappa)
        l2 = np.log2(fc); pi = np.minimum(1.0, KAPPA * np.abs(l2))
        part = (pi < 1.0) & (l2 != 0)
        fc = np.where(part, 2.0 ** (np.sign(l2) / KAPPA), fc)
        hit = rng.random(X0.shape) < np.where(part, pi, 1.0)[None, :]
        down = np.flatnonzero(fc < 1)
        if len(down):
            Xd = X0[:, down]; X[:, down] = np.where(hit[:, down], rng.binomial(Xd, fc[down][None, :]), Xd)
        up = np.flatnonzero(fc > 1)
        if len(up):
            X[:, up] += np.where(hit[:, up], rng.poisson(lib[:, None] * rate[None, up] * (fc[up] - 1)[None, :]), 0)
        mode = "dual"
        fb = 2.0 ** np.clip(np.nan_to_num(lfc_cells if lfc_bulk is None else lfc_bulk), -LFC_CLIP, LFC_CLIP)
        T = np.rint(X.shape[0] * ref_mean * fb).astype(np.int64)
        _fix_dual(X, T, TOP_FRAC, int(fix_seed))
        out = sp.csr_matrix(X.astype(np.float32)); out.eliminate_zeros()
        return out
    down = fc < 1
    if down.any():
        if os.environ.get("DOWN_HET") == "1":       # 2026-10-23: a down gene is lost completely in a share 1 - fc of the cells
            keep = rng.random((X0.shape[0], int(down.sum()))) < fc[down][None, :]      # and untouched in the others (real
            X[:, down] = np.where(keep, X0[:, down], 0)                                # down-regulation is heterogeneous)
        else:
            X[:, down] = rng.binomial(X0[:, down], fc[down][None, :])
    up = np.flatnonzero(fc > 1)
    if len(up):
        X[:, up] += rng.poisson(lib[:, None] * rate[None, up] * (fc[up] - 1)[None, :])
    if mode != "current":
        fb = fc if lfc_bulk is None else 2.0 ** np.clip(np.nan_to_num(lfc_bulk), -LFC_CLIP, LFC_CLIP)
        T = np.rint(X.shape[0] * ref_mean * fb).astype(np.int64)
        if mode == "exact":
            _fix_exact(X, T, np.cumsum(lib.astype(np.float64)), int(fix_seed))
        elif mode == "dual":
            _fix_dual(X, T, TOP_FRAC, int(fix_seed))
        elif mode == "dual_lib":
            _fix_dual_lib(X, T, lib.astype(np.float64), 0.10, 0.25, int(fix_seed))
        else:
            raise ValueError(mode)
    out = sp.csr_matrix(X.astype(np.float32)); out.eliminate_zeros()
    return out
