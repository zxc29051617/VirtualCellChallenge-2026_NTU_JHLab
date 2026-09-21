"""Does a second measurement of the same knockdown denoise the transfer?

The perturbation-specific component of a response transfers between Replogle's own two cell
lines at r = 0.024 (findings section 7). That is weak, but it is the whole of what earns
discrimination, which is 41% of the leader's score. If the weakness is noise rather than
genuine context-specificity, averaging two independent measurements of the same knockdown
should help; if it is context-specificity, averaging will not.

This decides it by measurement. RPE1 is held out: only its control profile and the target
list are used, which is exactly what the Challenge hands you for a scored context. The
predictions differ only in their source.

    source A   K562 genome-wide (Replogle)
    source B   K562 genome-wide averaged with CD4 T cells (Marson 2025)

CD4's fold changes come from a different pipeline, so their scale is calibrated against
gwps by least squares over the shared entries. That calibration reads only the two source
datasets -- never RPE1 -- so it is available at submission time too.

    python scripts/compare_cd4_transfer.py
    python scripts/compare_cd4_transfer.py --predict K562 --source RPE1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vcc2026.local_eval import fold_change, load_profiles, means_from_fold_change, score

CONTROL = "control"
PROFILES = {
    "K562gw": "external/profiles/K562gw_perturbation_means.npz",
    "K562": "external/profiles/K562_perturbation_means.npz",
    "RPE1": "external/profiles/RPE1_perturbation_means.npz",
}


def _load_cd4(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Per-target log fold change, averaged over the culture conditions."""
    payload = np.load(path, allow_pickle=True)
    log_fc = payload["log_fc"]
    targets = [str(t) for t in payload["targets"]]
    genes = [str(g) for g in payload["genes"]]

    rows: dict[str, list[int]] = {}
    for index, target in enumerate(targets):
        rows.setdefault(target, []).append(index)
    # Replicates of one knockdown in different media: averaging them is the same denoising
    # argument one level down, and it costs nothing.
    effects = {
        target: np.nan_to_num(np.nanmean(log_fc[index, :], axis=0))
        for target, index in rows.items()
    }
    return effects, genes


def _calibrate(source: np.ndarray, other: np.ndarray) -> float:
    """Least-squares scale putting `other` in `source`'s units, through the origin.

    Absorbs the log base and any difference in effect size between the two pipelines. Uses
    only source data, so the same number is available for a context with no answer key.
    """
    denominator = float((other**2).sum())
    return float((source * other).sum() / denominator) if denominator > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", default="RPE1", choices=sorted(PROFILES))
    parser.add_argument("--source", default="K562gw", choices=sorted(PROFILES))
    parser.add_argument("--cd4", default="external/profiles/CD4_target_lfc.npz")
    parser.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    target = load_profiles(PROFILES[args.predict])
    source = load_profiles(PROFILES[args.source])
    cd4_effects, cd4_genes = _load_cd4(Path(args.cd4))

    # One gene axis for everything, in the held-out context's order.
    source_columns = {gene: i for i, gene in enumerate(source.genes)}
    cd4_columns = {gene: i for i, gene in enumerate(cd4_genes)}
    shared = [g for g in target.genes if g in source_columns]
    target_take = np.array([target.genes.index(g) for g in shared])
    source_take = np.array([source_columns[g] for g in shared])
    cd4_take = np.array([cd4_columns.get(g, -1) for g in shared])
    cd4_known = cd4_take >= 0

    # Perturbations all three can speak to, so A and B are scored on the same rows.
    scored = [
        p for p in target.perturbations
        if p != CONTROL and p in source.perturbations and p in cd4_effects
    ]
    print(f"{len(shared):,} shared genes, {len(scored):,} shared perturbations")
    print(f"CD4 covers {int(cd4_known.sum()):,} of them\n")
    if not scored:
        raise SystemExit("no perturbation is present in all three datasets")

    target_rows = np.array([target.index_of(p) for p in scored])
    source_rows = np.array([source.index_of(p) for p in scored])

    target_control = target.means[target.index_of(CONTROL)][target_take]
    source_control = source.means[source.index_of(CONTROL)][source_take]
    measured = target.means[np.ix_(target_rows, target_take)]

    source_lfc = fold_change(source.means[np.ix_(source_rows, source_take)], source_control)
    cd4_lfc = np.zeros_like(source_lfc)
    for row, perturbation in enumerate(scored):
        effect = cd4_effects[perturbation]
        cd4_lfc[row, cd4_known] = effect[cd4_take[cd4_known]]

    scale = _calibrate(source_lfc[:, cd4_known], cd4_lfc[:, cd4_known])
    cd4_lfc *= scale
    agreement = np.corrcoef(
        source_lfc[:, cd4_known].ravel(), cd4_lfc[:, cd4_known].ravel()
    )[0, 1]
    print(f"CD4 scale against {args.source}: {scale:.3f}   correlation r = {agreement:.4f}")
    print("(r near 0 would mean the two sources share almost no signal to average)\n")

    header = f"{'CD4 weight':>11}  {'pds':>7}  {'expr_error':>10}  {'direction':>9}  {'jaccard':>7}"
    print(header)
    print("-" * len(header))
    for weight in [float(w) for w in args.weights.split(",")]:
        blended = source_lfc.copy()
        # Genes CD4 never measured keep the single source; blending in a zero there would
        # shrink real effects toward nothing and look like a win on expression error alone.
        blended[:, cd4_known] = (
            (1.0 - weight) * source_lfc[:, cd4_known] + weight * cd4_lfc[:, cd4_known]
        )
        predicted = means_from_fold_change(blended, target_control)
        result = score(
            predicted, measured, target_control, perturbations=scored, genes=shared
        )
        print(f"{weight:>11.2f}  {result['pds']:>7.4f}  {result['expr_error']:>10.4f}"
              f"  {result['direction']:>9.4f}  {result['jaccard']:>7.4f}")

    print("\nweight 0.00 is the current V3 transfer. Higher pds and direction are better;"
          "\nlower expr_error is better. A gain has to survive the reverse fold too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
