"""Truth-free check: how much cell-level change does dual rendering keep, by gene expression level?
Only control cells and our predictions are used. For each sampled target, per-gene cell-level mean-log shift
(the quantity behind the DE fold change) under dual (bulk trans x1, cells x3) relative to exact x3 (V1),
binned by the gene's mean count per control cell."""
import sys, json, numpy as np
sys.path.insert(0, "."); import render_eval as E
from vcc.render import render
BINS = [0, 0.1, 0.3, 1, 3, 10, 1e9]
out = {}
for name in ["H1", "KOLF200", "HCT116"]:
    D, R, C, lib, rate, ref_mean = E.load_set(name); targets, oc, full, shuf = E.components(name)
    cn = C.multiply(1e4 / lib[:, None]).tocsr(); cn.data = np.log1p(cn.data); ctrl_ml = np.asarray(cn.mean(0)).ravel()
    def ml(Y):
        yn = Y.multiply(1e4 / np.asarray(Y.sum(1))).tocsr(); yn.data = np.log1p(yn.data); return np.asarray(yn.mean(0)).ravel()
    ratios = {b: [] for b in range(len(BINS) - 1)}; binof = np.digitize(ref_mean, BINS) - 1
    for i in range(0, len(targets), max(1, len(targets) // 20)):
        tr = np.abs(full[i] - oc[i]) > 0.3
        draw = np.random.default_rng([11, i, 0]).choice(C.shape[0], E.N_PRED, replace=False)
        a = render(C[draw], lib[draw], rate, ref_mean, full[i], np.random.default_rng([11, i, 1]), mode="exact", fix_seed=i)
        b = render(C[draw], lib[draw], rate, ref_mean, full[i], np.random.default_rng([11, i, 1]), mode="dual", lfc_bulk=oc[i] + (full[i] - oc[i]) / 3, fix_seed=i)
        sa = (ml(a) - ctrl_ml) * np.sign(full[i]); sb = (ml(b) - ctrl_ml) * np.sign(full[i])
        for k in range(len(BINS) - 1):
            m = tr & (binof == k) & (sa > 1e-4)
            if m.sum() >= 5: ratios[k].append(float(np.median(sb[m] / sa[m])))
    out[name] = {f"{BINS[k]}-{BINS[k+1]}": (round(float(np.median(v)), 3) if v else None, len(v)) for k, v in ratios.items()}
    print(name, out[name], flush=True)
json.dump(out, open("retention_by_bin.json", "w"), indent=1)
