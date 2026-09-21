"""Probe 2: (i) negative lambda -- sharpening away from the class mean;
(ii) what to give targets the source screen does not cover.

Same K562 -> RPE1 split as scripts/evaluate_local.py. For (ii) 11% of the scored targets
(33/300, the Challenge's uncovered fraction) have their source measurement discarded and are
filled with either the global mean response or their own k-means class mean -- an ORACLE
class assignment, since a real uncovered target has no source lfc to cluster on, so this is
an upper bound on what GO-based class fill could buy.

Throwaway probe; not part of the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.local_eval import (  # noqa: E402
    _own_gene_columns,
    fold_change,
    load_profiles,
    log_profiles,
    means_from_fold_change,
    score,
)

CONTROL = "control"
ROOT = Path("D:/VirtualCellChallenge/external/profiles")
MIN_CELLS = 25
SEED = 0


def per_row_pds(pred_means, measured, control, perts, genes):
    own = _own_gene_columns(perts, genes)
    clog = log_profiles(control.reshape(1, -1))
    p = log_profiles(pred_means) - clog
    t = log_profiles(measured) - clog
    keep = np.ones(p.shape[1], dtype=bool)
    keep[own[own >= 0]] = False
    p, t = p[:, keep], t[:, keep]
    pn = np.linalg.norm(p, axis=1, keepdims=True)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    d = 1.0 - (p / np.where(pn > 0, pn, 1.0)) @ (t / np.where(tn > 0, tn, 1.0)).T
    d[(pn == 0).ravel(), :] = 1.0
    d[:, (tn == 0).ravel()] = 1.0
    n = d.shape[0]
    own_d = np.diagonal(d)[:, None]
    closer = (d < own_d).sum(axis=1)
    tied = (d == own_d).sum(axis=1) - 1
    return 1.0 - (closer + 0.5 * tied) / (n - 1)


def main() -> int:
    target = load_profiles(ROOT / "RPE1_perturbation_means.npz")
    source = load_profiles(ROOT / "K562_perturbation_means.npz")
    genes = [g for g in target.genes if g in set(source.genes)]
    tcol = [target.genes.index(g) for g in genes]
    scol = [source.genes.index(g) for g in genes]
    shared = set(source.perturbations) & set(target.perturbations)
    usable = [
        p for p in target.perturbations
        if p in shared and p != CONTROL and target.counts[target.index_of(p)] >= MIN_CELLS
    ]
    order = np.random.default_rng(SEED).permutation(len(usable))
    score_rows = order[len(usable) // 2:]
    scored = [usable[i] for i in score_rows]
    tctrl = target.means[target.index_of(CONTROL), tcol]
    sctrl = source.means[source.index_of(CONTROL), scol]
    measured = target.means[np.ix_([target.index_of(p) for p in usable], tcol)][score_rows]
    smeans = source.means[np.ix_([source.index_of(p) for p in usable], scol)]
    source_lfc = fold_change(smeans, sctrl)

    def report(name, lfc_scored):
        pred = means_from_fold_change(lfc_scored, tctrl)
        s = score(np.ascontiguousarray(pred), measured, tctrl,
                  perturbations=scored, genes=genes, top_n=100)
        print(f"{name:<34}{s['pds']:>9.4f}{s['expr_error']:>13.4f}"
              f"{s['direction']:>11.4f}{s['jaccard']:>10.4f}")
        return pred

    print(f"{'model':<34}{'pds':>9}{'expr_error':>13}{'direction':>11}{'jaccard':>10}")
    print("-" * 77)
    report("transfer (lambda=0)", source_lfc[score_rows])

    print("\n(i) sharpening: lambda < 0 pushes each target AWAY from its class mean")
    for k in (5, 20, 200):
        norm = np.linalg.norm(source_lfc, axis=1, keepdims=True)
        _, labels = kmeans2(source_lfc / np.where(norm > 0, norm, 1.0), k,
                            minit="++", seed=SEED, iter=50, missing="warn")
        for lam in (-0.25, -0.5):
            out = source_lfc.copy()
            for c in np.unique(labels):
                rows = labels == c
                out[rows] = (1 - lam) * source_lfc[rows] + lam * source_lfc[rows].mean(axis=0)
            report(f"kmeans K={k} lambda={lam}", out[score_rows])
    grand = source_lfc.mean(axis=0)
    for lam in (-0.25, -0.5, -1.0):
        report(f"global mean removed x{-lam}", (source_lfc - lam * -1 * grand)[score_rows]
               if False else (source_lfc + lam * grand)[score_rows])

    print("\n(ii) targets the source screen does not cover (11% of the panel)")
    rng = np.random.default_rng(7)
    n_hidden = int(round(0.11 * len(scored)))
    hidden = rng.choice(len(scored), size=n_hidden, replace=False)
    hidden_mask = np.zeros(len(scored), dtype=bool)
    hidden_mask[hidden] = True
    base = source_lfc[score_rows].copy()

    variants = {"global mean response": np.broadcast_to(grand, (n_hidden, base.shape[1]))}
    norm = np.linalg.norm(source_lfc, axis=1, keepdims=True)
    for k in (5, 20, 60):
        _, labels = kmeans2(source_lfc / np.where(norm > 0, norm, 1.0), k,
                            minit="++", seed=SEED, iter=50, missing="warn")
        cmeans = np.stack([source_lfc[labels == c].mean(axis=0) for c in range(labels.max() + 1)])
        variants[f"ORACLE class mean K={k}"] = cmeans[labels[score_rows][hidden_mask]]

    print(f"  {n_hidden} of {len(scored)} scored targets hidden")
    for name, fill in variants.items():
        lfc = base.copy()
        lfc[hidden_mask] = fill
        pred = report(f"  fill = {name}", lfc)
        rows = per_row_pds(pred, measured, tctrl, scored, genes)
        print(f"{'':<34}{'':>9}   hidden-row pds {rows[hidden_mask].mean():.4f}"
              f" | covered-row pds {rows[~hidden_mask].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
