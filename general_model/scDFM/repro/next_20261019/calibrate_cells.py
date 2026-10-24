"""Internal calibration (no truth used): which cell-level trans amplitude makes dual-rendered cells show the same
cell-level change (median per-cell mean-log shift on trans genes, the quantity behind the DE fold change) as exact
rendering at amplitude 3? Uses only a context's control cells and our predictions."""
import sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, "."); import render_eval as E
from vcc.render import render
name = sys.argv[1] if len(sys.argv) > 1 else "H1"
D, R, C, lib, rate, ref_mean = E.load_set(name); targets, oc, full, shuf = E.components(name)
cn = C.multiply(1e4 / lib[:, None]).tocsr(); cn.data = np.log1p(cn.data); ctrl_ml = np.asarray(cn.mean(0)).ravel()
def shift(Y, sign, tr):
    yn = Y.multiply(1e4 / np.asarray(Y.sum(1))).tocsr(); yn.data = np.log1p(yn.data)
    return np.asarray(yn.mean(0)).ravel()[tr] - ctrl_ml[tr]
res = {}
for i in range(0, len(targets), max(1, len(targets) // 25)):
    tr = np.abs(full[i] - oc[i]) > 0.3
    if tr.sum() < 20: continue
    sign = np.sign(full[i][tr]); draw = np.random.default_rng([9, i, 0]).choice(C.shape[0], E.N_PRED, replace=False)
    ref = render(C[draw], lib[draw], rate, ref_mean, full[i], np.random.default_rng([9, i, 1]), mode="exact", fix_seed=i)
    r0 = shift(ref, sign, tr) * sign
    for a in [3, 4, 5, 6, 8]:
        lc = oc[i] + (full[i] - oc[i]) * (a / 3.0); lb = oc[i] + (full[i] - oc[i]) / 3.0
        Y = render(C[draw], lib[draw], rate, ref_mean, lc, np.random.default_rng([9, i, 1]), mode="dual", lfc_bulk=lb, fix_seed=i)
        res.setdefault(a, []).append(np.median(shift(Y, sign, tr) * sign) / max(np.median(r0), 1e-9))
print(name, {a: round(float(np.median(v)), 3) for a, v in res.items()}, "(cell-level shift relative to exact x3; 1.0 = matched)")
