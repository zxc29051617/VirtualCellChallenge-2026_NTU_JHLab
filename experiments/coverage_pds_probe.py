"""Probe 4: does dropping 70% of the scored genes to zero collapse pds to the floor?

The first real submission scored pds raw 0.501 while the same model scores 0.821 here. It
carried the top 3,000 genes by expression and sent the other 6,923 DE-scored genes as zero.
A sparse matrix's absent entries are zeros, so every predicted effect vector picks up the
same large negative offset -b_ctrl at those coordinates. That offset does not depend on the
perturbation, so every row of the distance matrix orders the 300 candidates the same way and
the correct one takes a uniform rank -- pds 0.5 by construction rather than by model failure.

This reproduces that here by censoring the local transfer prediction the same way.

Throwaway probe; not part of the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.local_eval import (  # noqa: E402
    CPM_TOTAL,
    LOW_EXPRESSION_CPM,
    _rescale,
    fold_change,
    load_profiles,
    means_from_fold_change,
    score,
)

CONTROL = "control"
ROOT = Path("D:/VirtualCellChallenge/external/profiles")


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
    order = np.random.default_rng(0).permutation(len(usable))
    score_rows = order[len(usable) // 2:]
    scored = [usable[i] for i in score_rows]
    tctrl = target.means[target.index_of(CONTROL), tcol]
    sctrl = source.means[source.index_of(CONTROL), scol]
    tmeans = target.means[np.ix_([target.index_of(p) for p in usable], tcol)]
    smeans = source.means[np.ix_([source.index_of(p) for p in usable], scol)]
    measured = tmeans[score_rows]
    slfc = fold_change(smeans, sctrl)

    ctrl_cpm = _rescale(tctrl.reshape(1, -1), CPM_TOTAL).ravel()
    admitted = np.flatnonzero(ctrl_cpm > LOW_EXPRESSION_CPM)
    print(f"{len(genes):,} genes, {admitted.size:,} pass the 5 CPM control filter")

    full = means_from_fold_change(slfc[score_rows] * 0.3, tctrl)

    print(f"\n{'carried genes':<36}{'coverage':>10}{'pds':>9}{'expr_err':>11}")
    print("-" * 66)
    by_expression = np.argsort(-tctrl)
    for n_carried in (len(genes), 5000, 3000, 2000, 1000):
        keep = np.zeros(len(genes), dtype=bool)
        keep[by_expression[:n_carried]] = True
        censored = np.where(keep, full, 0.0)
        cov = keep[admitted].mean()
        s = score(np.ascontiguousarray(censored), measured, tctrl,
                  perturbations=scored, genes=genes, top_n=100)
        label = "all" if n_carried >= len(genes) else f"top {n_carried:,} by expression"
        print(f"{label:<36}{cov:>9.1%}{s['pds']:>9.4f}{s['expr_error']:>11.4f}")

    # Cheapest-first over the admitted set, which is what the repo now does.
    cost = np.minimum(tctrl, 1.0)
    order2 = admitted[np.argsort(cost[admitted])]
    for budget in (3000.0, 7742.0):
        afford = np.cumsum(cost[order2]) <= budget
        keep = np.zeros(len(genes), dtype=bool)
        keep[order2[afford]] = True
        censored = np.where(keep, full, 0.0)
        s = score(np.ascontiguousarray(censored), measured, tctrl,
                  perturbations=scored, genes=genes, top_n=100)
        print(f"{f'cheapest-first, budget {budget:,.0f}':<36}"
              f"{keep[admitted].mean():>9.1%}{s['pds']:>9.4f}{s['expr_error']:>11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
