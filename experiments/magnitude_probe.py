"""Probe 3: is per-target response MAGNITUDE predictable, and is it a free lever?

Cosine distance ignores a per-row scale, so setting each target's response size separately is
(to first order) invisible to pds while moving every magnitude metric. This asks whether the
size of a target's response transfers across cell lines even though its direction barely does
(r = 0.024), and adds a local stand-in for de_wilcoxon_lfc_nmae.

Throwaway probe; not part of the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.local_eval import (  # noqa: E402
    CPM_TOTAL,
    LOW_EXPRESSION_CPM,
    _own_gene_columns,
    _rescale,
    fold_change,
    load_profiles,
    log2_fold_change,
    means_from_fold_change,
    score,
)

CONTROL = "control"
ROOT = Path("D:/VirtualCellChallenge/external/profiles")
SEED = 0


def nmae_proxy(pred_means, measured_means, control, perts, genes, top_n=100):
    """Local stand-in for de_wilcoxon_lfc_nmae: mean |err| / mean |true| over the gate,
    where the gate is the measured top movers among expressed genes (spec section 6 gates on
    the reference's own significant set)."""
    own = _own_gene_columns(perts, genes)
    expressed = _rescale(control.reshape(1, -1), CPM_TOTAL) > LOW_EXPRESSION_CPM
    true = log2_fold_change(measured_means, control)
    pred = log2_fold_change(pred_means, control)
    masked = true * expressed
    rows = np.arange(len(perts))
    ok = own >= 0
    masked[rows[ok], own[ok]] = 0.0
    cut = np.partition(np.abs(masked), -top_n, axis=1)[:, -top_n][:, None]
    gate = np.abs(masked) >= cut
    num = (np.abs(pred - true) * gate).sum(axis=1)
    den = (np.abs(true) * gate).sum(axis=1)
    return float(np.mean(num[den > 0] / den[den > 0]))


def main() -> int:
    target = load_profiles(ROOT / "RPE1_perturbation_means.npz")
    source = load_profiles(ROOT / "K562_perturbation_means.npz")
    genes = [g for g in target.genes if g in set(source.genes)]
    tcol = [target.genes.index(g) for g in genes]
    scol = [source.genes.index(g) for g in genes]
    shared = set(source.perturbations) & set(target.perturbations)
    usable = [
        p for p in target.perturbations
        if p in shared and p != CONTROL and target.counts[target.index_of(p)] >= 25
    ]
    order = np.random.default_rng(SEED).permutation(len(usable))
    fit_rows, score_rows = order[: len(usable) // 2], order[len(usable) // 2:]
    scored = [usable[i] for i in score_rows]
    tctrl = target.means[target.index_of(CONTROL), tcol]
    sctrl = source.means[source.index_of(CONTROL), scol]
    tmeans = target.means[np.ix_([target.index_of(p) for p in usable], tcol)]
    smeans = source.means[np.ix_([source.index_of(p) for p in usable], scol)]
    measured = tmeans[score_rows]

    slfc = fold_change(smeans, sctrl)
    tlfc = fold_change(tmeans, tctrl)
    snorm = np.linalg.norm(slfc, axis=1)
    tnorm = np.linalg.norm(tlfc, axis=1)

    print("Does response SIZE transfer across lines, where direction does not?")
    print(f"  pearson  r(||K562||, ||RPE1||) = {stats.pearsonr(snorm, tnorm)[0]:+.4f}")
    print(f"  spearman r                     = {stats.spearmanr(snorm, tnorm)[0]:+.4f}")
    print(f"  log-log pearson                = "
          f"{stats.pearsonr(np.log(snorm), np.log(tnorm))[0]:+.4f}")
    print(f"  (for comparison, direction transfers at r = 0.024)")

    pos = {g: i for i, g in enumerate(genes)}
    own = np.array([pos.get(p, -1) for p in usable])
    have = own >= 0
    tctrl_cpm = _rescale(tctrl.reshape(1, -1), CPM_TOTAL).ravel()
    print(f"\n  target-gene expression in the PREDICTED line's own control vs its response size")
    print(f"    spearman = "
          f"{stats.spearmanr(np.log1p(tctrl_cpm[own[have]]), tnorm[have])[0]:+.4f}"
          f"   ({have.sum()} of {len(usable)} targets measured)")

    norm = np.linalg.norm(slfc, axis=1, keepdims=True)
    _, labels = kmeans2(slfc / np.where(norm > 0, norm, 1.0), 20,
                        minit="++", seed=SEED, iter=50, missing="warn")
    cls_mean_norm = np.array([snorm[labels == c].mean() for c in range(labels.max() + 1)])
    print(f"    functional-class (kmeans K=20) mean ||K562|| vs ||RPE1||: spearman = "
          f"{stats.spearmanr(cls_mean_norm[labels], tnorm)[0]:+.4f}")

    def report(name, lfc):
        pred = means_from_fold_change(lfc, tctrl)
        s = score(np.ascontiguousarray(pred), measured, tctrl,
                  perturbations=scored, genes=genes, top_n=100)
        n = nmae_proxy(pred, measured, tctrl, scored, genes)
        print(f"{name:<38}{s['pds']:>8.4f}{s['expr_error']:>12.4f}"
              f"{n:>9.4f}{s['direction']:>11.4f}{s['jaccard']:>9.4f}")

    print(f"\n{'model':<38}{'pds':>8}{'expr_err':>12}{'nmae~':>9}{'direction':>11}{'jaccard':>9}")
    print("-" * 78)
    report("transfer, no scaling", slfc[score_rows])
    for s in (0.3, 0.5, 0.75):
        report(f"transfer x {s} (uniform)", slfc[score_rows] * s)

    # Per-target rescale to a PREDICTED norm. Fit the norm map on the fit half only.
    a, b = np.polyfit(np.log(snorm[fit_rows]), np.log(tnorm[fit_rows]), 1)
    want = np.exp(a * np.log(snorm) + b)
    factor = want / snorm
    print(f"\n  norm map fitted on the fit half: log||RPE1|| = {a:.3f} log||K562|| + {b:.3f}")
    print(f"  implied per-target scale: median {np.median(factor[score_rows]):.3f}, "
          f"range {factor[score_rows].min():.3f}-{factor[score_rows].max():.3f}")
    for extra in (1.0, 0.6, 0.4):
        report(f"per-target norm match x {extra}",
               slfc[score_rows] * (factor[score_rows, None] * extra))

    # Class-level scale only (what a GO partition could supply for an uncovered target).
    cls_factor = np.array([factor[labels == c].mean() for c in range(labels.max() + 1)])
    for extra in (1.0, 0.6):
        report(f"class-level norm match x {extra}",
               slfc[score_rows] * (cls_factor[labels[score_rows], None] * extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
