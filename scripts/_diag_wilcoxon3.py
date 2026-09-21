"""The real transfer prediction through the scorer's Wilcoxon test.

Uses the actual gwps-transferred fold changes at the shipped scale=0.3, on context A's
own control, and reports what each cell build would put in the DE table.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from vcc2026.io import read_csv_column, read_name_list
from vcc2026.transfer import source_fold_changes

from _diag_wilcoxon import (
    PATH, N_CELLS, N_PATTERNS, SEED, bh, cells_dirichlet_multinomial,
    cells_floor_ceil, load_control, wilcoxon_table,
)

ROOT = r"D:\VirtualCellChallenge"
SCALE = 0.3


def main() -> None:
    rng = np.random.default_rng(SEED)
    gene_names = read_name_list(rf"{ROOT}\dataset\gene_names.csv")
    targets = read_csv_column(rf"{ROOT}\dataset\pert_counts.csv", "target_gene")
    fc, covered = source_fold_changes(
        rf"{ROOT}\external\profiles\K562gw_perturbation_means.npz", targets, gene_names
    )
    print(f"targets {len(targets)}, covered by gwps {covered.sum()}")

    ctrl_sub, mean_counts = load_control(PATH, N_CELLS)
    ctrl = np.asarray(ctrl_sub.todense(), dtype=np.float64)
    cpm = mean_counts * (1e6 / mean_counts.sum())
    gate = np.flatnonzero(cpm >= 5.0)
    ctrl_g = ctrl[:, gate]
    base = mean_counts[gate]
    cm = ctrl_g.mean(axis=0)
    phi = np.where(cm > 0, ctrl_g.var(axis=0) / np.maximum(cm, 1e-12), 1.0)
    size = np.where(phi > 1.02, cm / np.maximum(phi - 1.0, 1e-6), 1e6)

    shifts = fc[:, gate] * SCALE                      # natural log
    live = np.flatnonzero(covered)
    a = np.abs(shifts[live])
    print(f"predicted |lfc| over the gate (natural log, scale={SCALE}): "
          f"median {np.median(a):.4f}  mean {a.mean():.4f}  p90 {np.percentile(a,90):.4f}  "
          f"sd {shifts[live].std():.4f}")
    print(f"  in log2: median {np.median(a)/np.log(2):.4f}  sd {shifts[live].std()/np.log(2):.4f}")

    print(f"\n{'build':>20} {'pert':>6} {'n_pred':>7} {'sig%':>6} "
          f"{'r(|z|,|lfc|)':>12} {'r(|z|,logmean)':>15}")
    for p in live[:6]:
        for kind, s in (("floorceil", 0.15), ("dirmult", 1.0)):
            mp = base * s * np.exp(shifts[p])
            pred = (cells_floor_ceil(mp, N_PATTERNS, N_CELLS, rng) if kind == "floorceil"
                    else cells_dirichlet_multinomial(mp, size, N_CELLS, rng))
            pv, auc, pm, cmn = wilcoxon_table(pred, ctrl_g)
            sig = bh(pv)
            zc = np.clip(stats.norm.isf(np.clip(pv, 1e-300, 1) / 2), 0, 40)
            print(f"{kind+' s='+format(s,'.2f'):>20} {targets[p]:>6} {sig.sum():7d} "
                  f"{100*sig.mean():6.1f} "
                  f"{np.corrcoef(zc, np.abs(shifts[p]))[0,1]:12.3f} "
                  f"{np.corrcoef(zc, np.log(base))[0,1]:15.3f}")


if __name__ == "__main__":
    main()
