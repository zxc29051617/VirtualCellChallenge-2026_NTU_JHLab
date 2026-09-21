"""Simulate the scorer's Wilcoxon rank-sum test against real control cells.

Chapter 3 of the metric spec: per gene, a two-sided Mann-Whitney U on CPM-normalized
per-cell values, prediction vs the reference's control cells, BH at alpha=0.05 over the
genes passing the 5-CPM control filter. Three of the six metrics read that table.

This asks what our own cell construction does to it when the PREDICTION IS NO CHANGE:
every call it produces is then a pure artifact of how integer cells are built.
"""
from __future__ import annotations

import sys
import h5py
import numpy as np
from scipy import sparse, stats

PATH = r"D:\VirtualCellChallenge\dataset\context_A.h5ad"
N_CELLS = 400
N_PATTERNS = 200
SEED = 0


def load_control(path: str, n_keep: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    with h5py.File(path, "r") as h:
        n_rows, n_genes = (int(x) for x in h["X"].attrs["shape"])
        indptr = h["X/indptr"][()]
        rng = np.random.default_rng(SEED)
        rows = np.sort(rng.choice(n_rows, size=n_keep, replace=False))
        data_parts, idx_parts, lens = [], [], []
        for r in rows:
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            idx_parts.append(h["X/indices"][lo:hi])
            data_parts.append(h["X/data"][lo:hi])
            lens.append(hi - lo)
        # full-set gene means, streamed
        gene_sums = np.zeros(n_genes, dtype=np.float64)
        for s in range(0, n_rows, 2048):
            e = min(n_rows, s + 2048)
            lo, hi = int(indptr[s]), int(indptr[e])
            gene_sums += np.bincount(
                h["X/indices"][lo:hi], weights=h["X/data"][lo:hi], minlength=n_genes
            )
        mean_counts = gene_sums / n_rows
    sub = sparse.csr_matrix(
        (np.concatenate(data_parts), np.concatenate(idx_parts),
         np.concatenate(([0], np.cumsum(lens)))),
        shape=(n_keep, n_genes),
    )
    return sub, mean_counts


def cells_floor_ceil(means: np.ndarray, n_patterns: int, n_cells: int,
                     rng: np.random.Generator) -> np.ndarray:
    """The repo's cells_from_mean: floor/ceil patterns, tiled to n_cells."""
    floor = np.floor(means)
    surplus = np.rint((means - floor) * n_patterns).astype(np.int64)
    cells = np.tile(floor, (n_patterns, 1))
    for g, c in enumerate(surplus):
        if c >= n_patterns:
            cells[:, g] += 1
        elif c > 0:
            cells[rng.choice(n_patterns, size=int(c), replace=False), g] += 1
    reps = int(np.ceil(n_cells / n_patterns))
    return np.tile(cells, (reps, 1))[:n_cells]


def cells_dirichlet_multinomial(means: np.ndarray, size: np.ndarray, n_cells: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Exact per-gene group total, allocated by Dirichlet-multinomial.

    (X_1..X_n) | sum = T for i.i.d. NB(r, p) is DirMult(T; r,...,r), so this reproduces
    the negative-binomial per-cell shape with the group mean held exactly.
    """
    out = np.zeros((n_cells, means.size), dtype=np.float64)
    totals = np.rint(means * n_cells).astype(np.int64)
    for g in range(means.size):
        T = int(totals[g])
        if T <= 0:
            continue
        a = max(float(size[g]), 1e-3)
        w = rng.gamma(a, 1.0, size=n_cells)
        s = w.sum()
        out[:, g] = rng.multinomial(T, w / s) if s > 0 else 0
    return out


def wilcoxon_table(pred: np.ndarray, ctrl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Mann-Whitney per gene on CPM values; returns (p, z-ish U-stat sign)."""
    pl = pred.sum(axis=1, keepdims=True)
    cl = ctrl.sum(axis=1, keepdims=True)
    pcpm = pred / np.where(pl > 0, pl, 1) * 1e6
    ccpm = ctrl / np.where(cl > 0, cl, 1) * 1e6
    res = stats.mannwhitneyu(pcpm, ccpm, axis=0, alternative="two-sided", method="asymptotic")
    n1, n2 = pred.shape[0], ctrl.shape[0]
    auc = res.statistic / (n1 * n2)          # P(pred > ctrl) + 0.5 P(tie)
    return res.pvalue, auc, pcpm.mean(axis=0), ccpm.mean(axis=0)


def bh(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    m = p.size
    order = np.argsort(p)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.flatnonzero(passed)
    out = np.zeros(m, dtype=bool)
    if k.size:
        out[order[: k[-1] + 1]] = True
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    ctrl_sub, mean_counts = load_control(PATH, N_CELLS)
    ctrl = np.asarray(ctrl_sub.todense(), dtype=np.float64)

    total = mean_counts.sum()
    cpm = mean_counts * (1e6 / total)
    gate = np.flatnonzero(cpm >= 5.0)
    print(f"control cells sampled {ctrl.shape[0]}, median depth {np.median(ctrl.sum(1)):.0f}, "
          f"mean depth {ctrl.sum(1).mean():.0f}")
    print(f"gate (>=5 CPM on full control): {gate.size} genes of {mean_counts.size}")
    m = mean_counts[gate]
    print(f"gate mean counts: min {m.min():.4f} median {np.median(m):.3f} max {m.max():.1f}")
    for thr in (0.5, 1, 2, 3.5, 5, 10):
        print(f"  gate genes with control mean >= {thr:>4} counts: "
              f"{(m >= thr).sum():5d} ({100*(m>=thr).mean():.1f}%)")

    # per-gene dispersion and size from the control cells themselves
    cm = ctrl[:, gate].mean(axis=0)
    cv = ctrl[:, gate].var(axis=0)
    phi = np.where(cm > 0, cv / np.maximum(cm, 1e-12), 1.0)
    size = np.where(phi > 1.02, cm / np.maximum(phi - 1.0, 1e-6), 1e6)
    print(f"measured dispersion var/mean over gate: median {np.median(phi):.3f} "
          f"mean {phi.mean():.3f}; zero fraction of control cells over gate "
          f"{(ctrl[:, gate] == 0).mean():.3f}")

    scales = [float(x) for x in sys.argv[1:]] or [0.15, 0.3, 0.5, 1.0]
    print("\n== NO-CHANGE prediction: every call below is a construction artifact ==")
    hdr = (f"{'build':>22} {'nnz/cell':>9} {'sig':>6} {'sig%':>6} {'down%':>6} "
           f"{'medAUC':>7} {'med|z|':>7} {'zerofrac':>8}")
    print(hdr)
    for s in scales:
        for kind in ("floorceil", "dirmult"):
            mp = mean_counts[gate] * s
            if kind == "floorceil":
                pred = cells_floor_ceil(mp, N_PATTERNS, N_CELLS, rng)
            else:
                pred = cells_dirichlet_multinomial(mp, size, N_CELLS, rng)
            # renormalize library size so CPM reflects the whole cell, not just the gate
            p, auc, pm, cmn = wilcoxon_table(pred, ctrl[:, gate])
            sig = bh(p)
            z = stats.norm.isf(np.clip(p, 1e-300, 1) / 2)
            down = (auc < 0.5) & sig
            mean_err = np.abs(pm - cmn) / np.maximum(cmn, 1e-9)
            print(f"{kind+' s='+format(s,'.2f'):>22} {(pred>0).sum(1).mean():9.0f} "
                  f"{sig.sum():6d} {100*sig.mean():6.1f} "
                  f"{100*down.sum()/max(sig.sum(),1):6.1f} {np.median(auc):7.3f} "
                  f"{np.median(z):7.2f} {(pred==0).mean():8.3f}"
                  f"   meanCPM rel.err med {np.median(mean_err):.2e}")


if __name__ == "__main__":
    main()
