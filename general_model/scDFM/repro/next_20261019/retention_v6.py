"""Truth-free: cell-level change kept when cells follow V1 (full trans) and gene totals follow V5 (shared trans
components removed). Median over trans genes of (V6 cell-level shift / V1 cell-level shift), sign-aligned to V1."""
import sys, numpy as np
sys.path.insert(0, "."); import render_eval as E
from vcc.render import render
for name in ["H1", "KOLF200", "HCT116"]:
    D, R, C, lib, rate, ref_mean = E.load_set(name); targets, oc, full, shuf = E.components(name)
    v5 = oc + E.remove_shared(full - oc)
    cn = C.multiply(1e4 / lib[:, None]).tocsr(); cn.data = np.log1p(cn.data); ctrl_ml = np.asarray(cn.mean(0)).ravel()
    def ml(Y):
        yn = Y.multiply(1e4 / np.asarray(Y.sum(1))).tocsr(); yn.data = np.log1p(yn.data); return np.asarray(yn.mean(0)).ravel()
    keep6, keep5 = [], []
    for i in range(0, len(targets), max(1, len(targets) // 20)):
        tr = np.abs(full[i] - oc[i]) > 0.3
        draw = np.random.default_rng([13, i, 0]).choice(C.shape[0], E.N_PRED, replace=False)
        a = render(C[draw], lib[draw], rate, ref_mean, full[i], np.random.default_rng([13, i, 1]), mode="exact", fix_seed=i)
        b = render(C[draw], lib[draw], rate, ref_mean, full[i], np.random.default_rng([13, i, 1]), mode="dual", lfc_bulk=v5[i], fix_seed=i)
        c = render(C[draw], lib[draw], rate, ref_mean, v5[i], np.random.default_rng([13, i, 1]), mode="exact", fix_seed=i)
        sg = np.sign(full[i]); sa = (ml(a) - ctrl_ml) * sg; m = tr & (sa > 1e-4)
        keep6.append(np.median(((ml(b) - ctrl_ml) * sg)[m] / sa[m])); keep5.append(np.median(((ml(c) - ctrl_ml) * sg)[m] / sa[m]))
    print(f"{name}: cell-level change kept vs V1 -- V6 {np.median(keep6):.3f}, V5 {np.median(keep5):.3f}", flush=True)
