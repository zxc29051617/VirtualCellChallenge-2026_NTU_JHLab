"""Sensitivity of the scorer's Wilcoxon table to a REAL predicted shift.

_diag_wilcoxon.py showed what each build calls when the prediction is no change.
This asks the opposite: with a genuine per-gene log fold change applied, how many
genes does each build call, and is the call set driven by the shift or by the build?

n_pred drives fid's coverage term min(1, n_pred/n_real) and the jaccard denominator.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from _diag_wilcoxon import (
    PATH, N_CELLS, N_PATTERNS, SEED, bh, cells_dirichlet_multinomial,
    cells_floor_ceil, load_control, wilcoxon_table,
)


def main() -> None:
    rng = np.random.default_rng(SEED)
    ctrl_sub, mean_counts = load_control(PATH, N_CELLS)
    ctrl = np.asarray(ctrl_sub.todense(), dtype=np.float64)
    cpm = mean_counts * (1e6 / mean_counts.sum())
    gate = np.flatnonzero(cpm >= 5.0)
    ctrl_g = ctrl[:, gate]
    base = mean_counts[gate]

    cm = ctrl_g.mean(axis=0)
    phi = np.where(cm > 0, ctrl_g.var(axis=0) / np.maximum(cm, 1e-12), 1.0)
    size = np.where(phi > 1.02, cm / np.maximum(phi - 1.0, 1e-6), 1e6)

    print(f"{'build':>20} {'sigma_lfc':>9} {'n_pred':>7} {'sig%':>6} "
          f"{'r(|z|,|lfc|)':>12} {'r(|z|,logmean)':>15} {'signAcc@top500':>15}")
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        lfc = rng.normal(0.0, sigma, size=base.size)  # natural log
        for kind, s in (("floorceil", 0.15), ("dirmult", 1.0)):
            mp = base * s * np.exp(lfc)
            pred = (cells_floor_ceil(mp, N_PATTERNS, N_CELLS, rng) if kind == "floorceil"
                    else cells_dirichlet_multinomial(mp, size, N_CELLS, rng))
            p, auc, pm, cmn = wilcoxon_table(pred, ctrl_g)
            sig = bh(p)
            z = stats.norm.isf(np.clip(p, 1e-300, 1) / 2)
            zc = np.clip(z, 0, 40)
            r_lfc = np.corrcoef(zc, np.abs(lfc))[0, 1]
            r_exp = np.corrcoef(zc, np.log(base))[0, 1]
            # sign agreement of the submission's own lfc with the intended shift,
            # over the 500 genes the build ranks most confident
            top = np.argsort(-zc)[:500]
            pl = np.log2((pm + 1e-9) / (cmn + 1e-9))
            acc = float(np.mean(np.sign(pl[top]) == np.sign(lfc[top])))
            print(f"{kind+' s='+format(s,'.2f'):>20} {sigma:9.2f} {sig.sum():7d} "
                  f"{100*sig.mean():6.1f} {r_lfc:12.3f} {r_exp:15.3f} {acc:15.3f}")
        print()


if __name__ == "__main__":
    main()
