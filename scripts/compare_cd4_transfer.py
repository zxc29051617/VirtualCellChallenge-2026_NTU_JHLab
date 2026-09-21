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


def _calibrate(source: np.ndarray, other: np.ndarray, how: str) -> float:
    """Scale putting `other` in `source`'s units. Reads only source data, never the answer key.

    Two of them, because they answer different questions. `rms` equalizes effect size, so a
    blend is an average of two measurements and nothing else. `slope` is the least-squares
    fit through the origin, which is the better predictor but shrinks hard when the two
    sources agree weakly -- regression attenuation -- so part of any gain it shows is just
    prediction pulled toward no effect, which lowers squared error on its own.
    """
    if how == "rms":
        spread = float(np.sqrt((other**2).mean()))
        return float(np.sqrt((source**2).mean()) / spread) if spread > 0 else 0.0
    denominator = float((other**2).sum())
    return float((source * other).sum() / denominator) if denominator > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", default="RPE1", choices=sorted(PROFILES))
    parser.add_argument("--source", default="K562gw", choices=sorted(PROFILES))
    parser.add_argument("--cd4", default="external/profiles/CD4_target_lfc.npz")
    parser.add_argument("--weights", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--calibration", default="rms", choices=["rms", "slope"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    target = load_profiles(PROFILES[args.predict])
    source = load_profiles(PROFILES[args.source])
    cd4_effects, cd4_genes = _load_cd4(Path(args.cd4))

    # One gene axis for everything, in the held-out context's order.
    source_columns = {gene: i for i, gene in enumerate(source.genes)}
    cd4_columns = {gene: i for i, gene in enumerate(cd4_genes)}
    target_columns = {gene: i for i, gene in enumerate(target.genes)}
    shared = [g for g in target.genes if g in source_columns]
    target_take = np.array([target_columns[g] for g in shared])
    source_take = np.array([source_columns[g] for g in shared])
    cd4_take = np.array([cd4_columns.get(g, -1) for g in shared])
    cd4_known = cd4_take >= 0

    # Perturbations all three can speak to, so A and B are scored on the same rows.
    source_rows_by_name = {p: i for i, p in enumerate(source.perturbations)}
    target_rows_by_name = {p: i for i, p in enumerate(target.perturbations)}
    scored = [
        p for p in target.perturbations
        if p != CONTROL and p in source_rows_by_name and p in cd4_effects
    ]
    print(f"{len(shared):,} shared genes, {len(scored):,} shared perturbations")
    print(f"CD4 covers {int(cd4_known.sum()):,} of them\n")
    if not scored:
        raise SystemExit("no perturbation is present in all three datasets")

    target_rows = np.array([target_rows_by_name[p] for p in scored])
    source_rows = np.array([source_rows_by_name[p] for p in scored])

    target_control = target.means[target_rows_by_name[CONTROL]][target_take]
    source_control = source.means[source_rows_by_name[CONTROL]][source_take]
    measured = target.means[np.ix_(target_rows, target_take)]

    source_lfc = fold_change(source.means[np.ix_(source_rows, source_take)], source_control)
    cd4_lfc = np.zeros_like(source_lfc)
    for row, perturbation in enumerate(scored):
        effect = cd4_effects[perturbation]
        cd4_lfc[row, cd4_known] = effect[cd4_take[cd4_known]]

    scale = _calibrate(source_lfc[:, cd4_known], cd4_lfc[:, cd4_known], args.calibration)
    cd4_lfc *= scale
    agreement = np.corrcoef(
        source_lfc[:, cd4_known].ravel(), cd4_lfc[:, cd4_known].ravel()
    )[0, 1]
    print(f"CD4 scale against {args.source}: {scale:.3f} ({args.calibration})"
          f"   correlation r = {agreement:.4f}")
    print("(r near 0 would mean the two sources share almost no signal to average)\n")

    # The control that makes the result mean something: the same CD4 numbers attached to the
    # wrong knockdowns. Magnitude, sparsity and gene-level structure all survive; only the
    # target-specific part is destroyed. A blend that beats weight 0 but does not beat this
    # gained nothing from knowing which gene was knocked down.
    rng = np.random.default_rng(args.seed)
    identity = np.arange(len(scored))
    permutation = rng.permutation(len(scored))
    while np.any(permutation == identity):
        # One rng, drawn fresh each time. Re-seeding here would re-apply the same shuffle and
        # could cycle without ever landing on a derangement.
        permutation = rng.permutation(len(scored))
    variants = {"CD4": cd4_lfc, "shuffled": cd4_lfc[permutation]}

    def blend(other: np.ndarray, weight: float) -> dict[str, float]:
        blended = source_lfc.copy()
        # Genes CD4 never measured keep the single source; blending in a zero there would
        # shrink real effects toward nothing and look like a win on expression error alone.
        blended[:, cd4_known] = (
            (1.0 - weight) * source_lfc[:, cd4_known] + weight * other[:, cd4_known]
        )
        return score(
            means_from_fold_change(blended, target_control),
            measured,
            target_control,
            perturbations=scored,
            genes=shared,
        )

    header = (f"{'second source':>14}  {'weight':>6}  {'pds':>7}  {'expr_error':>10}"
              f"  {'direction':>9}  {'jaccard':>7}")
    print(header)
    print("-" * len(header))
    for name, other in variants.items():
        for weight in [float(w) for w in args.weights.split(",")]:
            if weight == 0.0 and name != "CD4":
                continue  # weight 0 is the same prediction either way
            result = blend(other, weight)
            label = f"{args.source} only" if weight == 0.0 else name
            print(f"{label:>14}  {weight:>6.2f}  {result['pds']:>7.4f}"
                  f"  {result['expr_error']:>10.4f}  {result['direction']:>9.4f}"
                  f"  {result['jaccard']:>7.4f}")
        print()

    print("Higher pds and direction are better; lower expr_error is better. The gain that"
          "\ncounts is CD4 over shuffled at the same weight, not over weight 0.00, and it"
          "\nhas to survive the reverse fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
