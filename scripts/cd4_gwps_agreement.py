"""Does CD4 agree with K562 about a specific knockdown, or only about knockdowns in general?

This asks the denoising question directly on the Challenge's own targets, using only files
already on disk, and it does not need an answer key at all. Two independent measurements of
the same knockdown, in two contexts about as far apart as the Challenge's can be -- a chronic
myeloid leukemia line and primary CD4 T cells:

    matched      corr(gwps effect of knocking down X, CD4 effect of knocking down X)
    mismatched   the same against every OTHER target

If matched is no better than mismatched, CD4 knows only what all CRISPRi knockdowns do -- a
generic stress response -- and averaging it in cannot help discrimination, which is the 41%
of the leader's score that transfer currently earns. If matched wins, the perturbation-
specific component does survive a large context change, and r = 0.024 between Replogle's two
lines was measurement noise rather than genuine context-specificity.

The headline number is the same statistic the Challenge scores: is each prediction closest to
its own target? 0.5 is no target-specific information, 1.0 is perfect ranking. It is computed
here on fold changes rather than log profiles, so read it as a diagnostic, not a leaderboard
value -- but the null of 0.5 holds either way.

    python scripts/cd4_gwps_agreement.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vcc2026.local_eval import fold_change, load_profiles, perturbation_discrimination

CONTROL = "control"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="external/profiles/K562gw_perturbation_means.npz")
    parser.add_argument("--cd4", default="external/profiles/CD4_target_lfc.npz")
    args = parser.parse_args()

    source = load_profiles(args.source)
    payload = np.load(Path(args.cd4), allow_pickle=True)
    cd4_targets = [str(t) for t in payload["targets"]]
    cd4_genes = [str(g) for g in payload["genes"]]

    rows: dict[str, list[int]] = {}
    for index, target in enumerate(cd4_targets):
        rows.setdefault(target, []).append(index)
    cd4_effect = {
        target: np.nan_to_num(np.nanmean(payload["log_fc"][index, :], axis=0))
        for target, index in rows.items()
    }

    shared_targets = sorted(set(cd4_effect) & set(source.perturbations) - {CONTROL})
    source_columns = {gene: i for i, gene in enumerate(source.genes)}
    shared_genes = [g for g in cd4_genes if g in source_columns]
    print(f"{len(shared_targets):,} targets measured by both, over {len(shared_genes):,} genes")
    if len(shared_targets) < 3:
        raise SystemExit("too few shared targets to compare matched against mismatched")

    cd4_take = np.array([cd4_genes.index(g) for g in shared_genes])
    source_take = np.array([source_columns[g] for g in shared_genes])
    source_rows = np.array([source.index_of(t) for t in shared_targets])
    control = source.means[source.index_of(CONTROL)][source_take]

    source_lfc = fold_change(source.means[np.ix_(source_rows, source_take)], control)
    cd4_lfc = np.vstack([cd4_effect[t][cd4_take] for t in shared_targets])

    # Correlate every target against every other, then split the matrix on its diagonal.
    def centered(matrix: np.ndarray) -> np.ndarray:
        out = matrix - matrix.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm > 0, norm, 1.0)

    correlation = centered(cd4_lfc) @ centered(source_lfc).T
    diagonal = np.eye(len(shared_targets), dtype=bool)
    matched = correlation[diagonal]
    mismatched = correlation[~diagonal]

    print(f"\nmatched    r = {matched.mean():+.4f}  (sd {matched.std():.4f})")
    print(f"mismatched r = {mismatched.mean():+.4f}  (sd {mismatched.std():.4f})")
    # A one-sample comparison against the mismatched mean: how many standard errors apart.
    separation = (matched.mean() - mismatched.mean()) / (matched.std() / np.sqrt(len(matched)))
    print(f"separation = {separation:+.1f} standard errors")

    beaten = (matched[:, None] > correlation[~diagonal].reshape(len(matched), -1)).mean()
    print(f"a target's own match outranks {beaten:.1%} of the mismatches")

    own = np.array([shared_genes.index(t) if t in shared_genes else -1 for t in shared_targets])
    pds = perturbation_discrimination(cd4_lfc, source_lfc, own_columns=own)
    print(f"\ndiscrimination of CD4 against K562: {pds:.4f}   (0.5 = no target-specific signal)")
    print("\nAbove 0.5 means the perturbation-specific component survives a context change"
          "\nfrom a leukemia line to primary T cells, so a second source can denoise it."
          "\nAt 0.5 it does not, and blending CD4 in cannot help discrimination.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
