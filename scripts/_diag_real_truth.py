"""The local K562 -> RPE1 loop, re-scored through the submission's own write path.

evaluate_local.py scores a dense continuous profile over the 7,225 genes the two lines share.
The file that is actually uploaded carries a subset of genes and zeroes the rest. This applies
that truncation to the same prediction and the same real RPE1 answer key, so the gap between
0.812 and 0.501 can be attributed rather than guessed at.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.local_eval import fold_change, load_profiles, means_from_fold_change

TS = 5e4
ROOT = Path(__file__).resolve().parents[1]
CONTROL = "control"


def spec_effect(profiles: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    p = profiles / profiles.sum(axis=1, keepdims=True)
    c = ctrl / ctrl.sum()
    return np.log1p(TS * p) - np.log1p(TS * c)[None, :]


def pds(pred: np.ndarray, true: np.ndarray, drop: np.ndarray) -> float:
    keep = np.ones(pred.shape[1], dtype=bool)
    keep[drop[drop >= 0]] = False
    a, b = pred[:, keep], true[:, keep]
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    d = 1.0 - (a / np.where(an > 0, an, 1)) @ (b / np.where(bn > 0, bn, 1)).T
    d[(an == 0).ravel(), :] = 1.0
    d[:, (bn == 0).ravel()] = 1.0
    own = np.diagonal(d)[:, None]
    closer = (d < own).sum(axis=1)
    tied = (d == own).sum(axis=1) - 1
    return float((1.0 - (closer + 0.5 * tied) / (d.shape[0] - 1)).mean())


def main() -> None:
    target = load_profiles(ROOT / "external/profiles/RPE1_perturbation_means.npz")
    source = load_profiles(ROOT / "external/profiles/K562_perturbation_means.npz")

    src_genes = set(source.genes)
    genes = [g for g in target.genes if g in src_genes]
    tcol = [target.genes.index(g) for g in genes]
    scol = [source.genes.index(g) for g in genes]

    shared_perts = set(source.perturbations) & set(target.perturbations)
    usable = [
        p
        for p in target.perturbations
        if p in shared_perts and p != CONTROL and target.counts[target.index_of(p)] >= 25
    ]
    # match evaluate_local.py's scored half exactly
    order = np.random.default_rng(0).permutation(len(usable))
    scored_rows = order[int(len(usable) * 0.5):]
    scored = [usable[i] for i in scored_rows]

    t_ctrl = target.means[target.index_of(CONTROL), tcol].astype(np.float64)
    s_ctrl = source.means[source.index_of(CONTROL), scol].astype(np.float64)
    measured = target.means[np.ix_([target.index_of(p) for p in scored], tcol)].astype(np.float64)
    src_means = source.means[np.ix_([source.index_of(p) for p in scored], scol)].astype(np.float64)

    lfc = fold_change(src_means, s_ctrl)
    predicted = means_from_fold_change(lfc, t_ctrl)
    drop = np.array([genes.index(p) if p in genes else -1 for p in scored])

    truth = spec_effect(measured, t_ctrl)
    cpm = t_ctrl * (1e6 / t_ctrl.sum())

    print(f"{len(scored):,} scored perturbations, {len(genes):,} shared genes")
    print(f"{(cpm >= 5).sum():,} genes over 5 CPM in the RPE1 control\n")

    selections = {
        "every shared gene (what evaluate_local.py scores)": np.arange(len(genes)),
        "top 3,000 by expression (what was submitted)": np.argsort(-t_ctrl)[:3000],
        "the 5 CPM DE-scored set (what HEAD carries)": np.flatnonzero(cpm >= 5),
        "every gene the control expresses": np.flatnonzero(t_ctrl > 0),
    }
    for name, cols in selections.items():
        sub = np.zeros_like(predicted)
        sub[:, cols] = predicted[:, cols]
        eff = spec_effect(sub, t_ctrl)
        shared = eff.mean(axis=0)
        spec_norm = np.linalg.norm(eff - shared, axis=1).mean()
        print(
            f"{name}\n"
            f"  genes {cols.size:,}   |shared| {np.linalg.norm(shared):8.3f}"
            f"   mean |specific| {spec_norm:7.3f}"
            f"   ratio {np.linalg.norm(shared) / spec_norm:7.2f}x\n"
            f"  pds vs measured RPE1 = {pds(eff, truth, drop):.4f}"
            f"   after removing the shared offset = "
            f"{pds(eff - shared, truth, drop):.4f}\n"
        )


if __name__ == "__main__":
    main()
