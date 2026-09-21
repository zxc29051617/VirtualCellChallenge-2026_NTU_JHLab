"""Two follow-ups.

(1) HEAD still zeroes 8,610 of the 18,533 genes, leaving a shared offset whose norm is 1.6x
    the mean perturbation-specific norm. On Replogle that ratio is 0.24x at full coverage and
    pds is 0.82. This injects offsets of a given ratio into the Replogle prediction and reads
    pds off, so the residual can be priced instead of assumed harmless.

(2) The Wilcoxon test the four DE metrics are built on reads per-cell counts, not the group
    mean. This measures how far the written cells' per-gene detection rate sits from the real
    control's, which is what a rank-sum test on zero-inflated counts actually compares.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.local_eval import fold_change, load_profiles, means_from_fold_change

TS = 5e4
ROOT = Path(__file__).resolve().parents[1]


def spec_effect(profiles, ctrl):
    p = profiles / profiles.sum(axis=1, keepdims=True)
    c = ctrl / ctrl.sum()
    return np.log1p(TS * p) - np.log1p(TS * c)[None, :]


def pds(pred, true, drop):
    keep = np.ones(pred.shape[1], dtype=bool)
    keep[drop[drop >= 0]] = False
    a, b = pred[:, keep], true[:, keep]
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    d = 1.0 - (a / np.where(an > 0, an, 1)) @ (b / np.where(bn > 0, bn, 1)).T
    own = np.diagonal(d)[:, None]
    closer = (d < own).sum(axis=1)
    tied = (d == own).sum(axis=1) - 1
    return float((1.0 - (closer + 0.5 * tied) / (d.shape[0] - 1)).mean())


def part_one() -> None:
    target = load_profiles(ROOT / "external/profiles/RPE1_perturbation_means.npz")
    source = load_profiles(ROOT / "external/profiles/K562_perturbation_means.npz")
    src = set(source.genes)
    genes = [g for g in target.genes if g in src]
    tcol = [target.genes.index(g) for g in genes]
    scol = [source.genes.index(g) for g in genes]
    shared = set(source.perturbations) & set(target.perturbations)
    usable = [
        p for p in target.perturbations
        if p in shared and p != "control" and target.counts[target.index_of(p)] >= 25
    ]
    rows = np.random.default_rng(0).permutation(len(usable))[int(len(usable) * 0.5):]
    scored = [usable[i] for i in rows]

    t_ctrl = target.means[target.index_of("control"), tcol].astype(np.float64)
    s_ctrl = source.means[source.index_of("control"), scol].astype(np.float64)
    measured = target.means[np.ix_([target.index_of(p) for p in scored], tcol)].astype(np.float64)
    smeans = source.means[np.ix_([source.index_of(p) for p in scored], scol)].astype(np.float64)

    pred = means_from_fold_change(fold_change(smeans, s_ctrl), t_ctrl)
    truth = spec_effect(measured, t_ctrl)
    drop = np.array([genes.index(p) if p in genes else -1 for p in scored])

    eff = spec_effect(pred, t_ctrl)
    base_shared = eff.mean(axis=0)
    spec = eff - base_shared
    spec_norm = np.linalg.norm(spec, axis=1).mean()

    # A realistic offset direction: what zeroing a set of genes actually does is subtract the
    # control's own log profile on those coordinates, so use that shape rather than noise.
    ctrl_log = np.log1p(TS * t_ctrl / t_ctrl.sum())
    direction = -ctrl_log / np.linalg.norm(ctrl_log)

    print("pds against measured RPE1 as a shared offset is added to every perturbation")
    print(f"  mean |specific| = {spec_norm:.3f}\n")
    print(f"  {'|shared|/|specific|':>22}  {'pds':>8}")
    for ratio in [0.0, 0.24, 0.5, 1.0, 1.6, 3.0, 7.35, 20.0, 50.0]:
        offset = direction * (ratio * spec_norm)
        print(f"  {ratio:>22.2f}  {pds(spec + offset, truth, drop):>8.4f}")


def part_two() -> None:
    path = ROOT / "dataset/context_A.h5ad"
    with h5py.File(path, "r") as h:
        n_rows, n_genes = (int(x) for x in h["X"].attrs["shape"])
        indptr = h["X/indptr"][()]
        per_cell = np.diff(indptr)
        lo, hi = int(indptr[0]), int(indptr[min(n_rows, 4000)])
        idx = h["X/indices"][lo:hi]
        dat = h["X/data"][lo:hi]
    detect = np.bincount(idx, minlength=n_genes) / min(n_rows, 4000)
    mean_counts = np.bincount(idx, weights=dat, minlength=n_genes) / min(n_rows, 4000)
    print("\n\nreal control cells, context A")
    print(f"  genes detected per cell: mean {per_cell.mean():,.0f} of {n_genes:,}"
          f"  ({100 * (1 - per_cell.mean() / n_genes):.1f}% zeros)")

    cpm = mean_counts * (1e6 / mean_counts.sum())
    admitted = cpm >= 5
    print(f"  over the {admitted.sum():,} DE-scored genes, detection rate:"
          f" median {np.median(detect[admitted]):.4f}")

    # what the writer produces for the same genes at profile_scale 0.15
    for ps in (1.0, 0.15):
        pred_mean = mean_counts * ps
        pred_detect = np.minimum(pred_mean, 1.0)
        ratio = pred_detect[admitted] / np.maximum(detect[admitted], 1e-12)
        print(f"  written cells at profile_scale {ps}: predicted detection rate /"
              f" control detection rate, median {np.median(ratio):.3f},"
              f" {100 * np.mean(ratio < 0.9):.1f}% of scored genes below 0.9")
    entries = np.minimum(mean_counts * 0.15, 1.0)[admitted].sum()
    print(f"  written cells carry {entries:,.0f} nonzero entries over those genes,"
          f" against {per_cell.mean():,.0f} in a real control cell")


if __name__ == "__main__":
    part_one()
    part_two()
