"""Probe: does shrinking each target's transferred effect toward a functional-class mean help?

Scores K562 -> RPE1 on Replogle with the same split as scripts/evaluate_local.py, sweeping
the shrinkage weight lambda in

    lfc_p  <-  (1 - lambda) * lfc_p  +  lambda * mean(lfc over p's class)

for several class granularities. Classes are k-means clusters of the SOURCE line's own log
fold changes, which is the best case for within-class coherence -- a GO partition can only
be a noisier version of it. A size-matched random partition is the negative control.

Throwaway probe; not part of the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.local_eval import (  # noqa: E402
    fold_change,
    load_profiles,
    means_from_fold_change,
    score,
)

CONTROL = "control"
ROOT = Path("D:/VirtualCellChallenge/external/profiles")
MIN_CELLS = 25
SEED = 0


def classes_kmeans(lfc: np.ndarray, k: int, seed: int) -> np.ndarray:
    norm = np.linalg.norm(lfc, axis=1, keepdims=True)
    unit = lfc / np.where(norm > 0, norm, 1.0)
    _, labels = kmeans2(unit, k, minit="++", seed=seed, iter=50, missing="warn")
    return labels


def classes_random(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, k, size=n)


def shrink(lfc: np.ndarray, labels: np.ndarray, lam: float) -> np.ndarray:
    out = lfc.copy()
    for c in np.unique(labels):
        rows = labels == c
        out[rows] = (1.0 - lam) * lfc[rows] + lam * lfc[rows].mean(axis=0)
    return out


def main() -> int:
    target = load_profiles(ROOT / "RPE1_perturbation_means.npz")
    source = load_profiles(ROOT / "K562_perturbation_means.npz")

    genes = [g for g in target.genes if g in set(source.genes)]
    tcol = [target.genes.index(g) for g in genes]
    scol = [source.genes.index(g) for g in genes]

    shared = set(source.perturbations) & set(target.perturbations)
    usable = [
        p
        for p in target.perturbations
        if p in shared and p != CONTROL and target.counts[target.index_of(p)] >= MIN_CELLS
    ]
    order = np.random.default_rng(SEED).permutation(len(usable))
    n_fit = len(usable) // 2
    score_rows = order[n_fit:]
    scored = [usable[i] for i in score_rows]

    tctrl = target.means[target.index_of(CONTROL), tcol]
    sctrl = source.means[source.index_of(CONTROL), scol]
    measured = target.means[np.ix_([target.index_of(p) for p in usable], tcol)][score_rows]
    smeans = source.means[np.ix_([source.index_of(p) for p in usable], scol)]

    source_lfc = fold_change(smeans, sctrl)
    target_lfc_all = fold_change(
        target.means[np.ix_([target.index_of(p) for p in usable], tcol)], tctrl
    )

    print(f"{len(usable):,} perturbations, {len(genes):,} genes, {len(scored):,} scored\n")

    def run(name: str, labels: np.ndarray | None, lam: float) -> None:
        lfc = source_lfc if labels is None else shrink(source_lfc, labels, lam)
        pred = means_from_fold_change(lfc[score_rows], tctrl)
        s = score(
            np.ascontiguousarray(pred), measured, tctrl,
            perturbations=scored, genes=genes, top_n=100,
        )
        print(
            f"{name:<26}{s['pds']:>9.4f}{s['expr_error']:>13.4f}"
            f"{s['direction']:>11.4f}{s['jaccard']:>10.4f}"
        )

    print(f"{'model':<26}{'pds':>9}{'expr_error':>13}{'direction':>11}{'jaccard':>10}")
    print("-" * 69)
    run("transfer (lambda=0)", None, 0.0)

    for k in (2, 5, 20, 60, 200):
        labels = classes_kmeans(source_lfc, k, SEED)
        sizes = np.bincount(labels)
        between = _between_fraction(source_lfc, labels)
        print(f"  -- kmeans K={k}: sizes {sizes.min()}-{sizes.max()}, "
              f"between-class variance {between:.3f}")
        for lam in (0.25, 0.5, 1.0):
            run(f"kmeans K={k} lambda={lam}", labels, lam)

    labels = classes_random(len(usable), 2, SEED)
    print("  -- random 2-way partition (negative control)")
    for lam in (0.5, 1.0):
        run(f"random K=2 lambda={lam}", labels, lam)

    # How much of the class structure survives the move to the other line?
    print("\ncross-line transfer of each component (Pearson r over all gene x pert entries)")
    grand = source_lfc.mean(axis=0)
    tgrand = target_lfc_all.mean(axis=0)
    for k in (2, 5, 20, 60):
        labels = classes_kmeans(source_lfc, k, SEED)
        s_class = np.zeros_like(source_lfc)
        t_class = np.zeros_like(target_lfc_all)
        for c in np.unique(labels):
            rows = labels == c
            s_class[rows] = source_lfc[rows].mean(axis=0) - grand
            t_class[rows] = target_lfc_all[rows].mean(axis=0) - tgrand
        s_spec = source_lfc - grand - s_class
        t_spec = target_lfc_all - tgrand - t_class
        print(
            f"  K={k:<4} class-component r={_corr(s_class, t_class):+.4f}   "
            f"residual r={_corr(s_spec, t_spec):+.4f}"
        )
    print(f"  grand-mean (shared) r={_corr(grand, tgrand):+.4f}")
    print(f"  full specific  r={_corr(source_lfc - grand, target_lfc_all - tgrand):+.4f}")
    return 0


def _between_fraction(lfc: np.ndarray, labels: np.ndarray) -> float:
    grand = lfc.mean(axis=0)
    centred = lfc - grand
    total = float((centred ** 2).sum())
    between = 0.0
    for c in np.unique(labels):
        rows = labels == c
        between += float(rows.sum()) * float(((lfc[rows].mean(axis=0) - grand) ** 2).sum())
    return between / total


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel(), b.ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
