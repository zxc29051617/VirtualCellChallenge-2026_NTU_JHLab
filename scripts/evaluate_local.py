"""Historical continuous-profile diagnostics, including target-supervised calibration.

For source-only predictions scored AFTER actual integer H5AD output/readback, use
scripts/offline_validation.py roundtrip and scripts/compare_generative_pilot.py.
This script does not exercise serialization. Its scaled/calibrated models use target
perturbations for fitting; mean_response is an oracle using the scored answer key.

Fit on K562, predict RPE1. RPE1's perturbation profiles are the answer key and are never
shown to a model -- only its control profile and the list of genes to knock down, which is
exactly what the Challenge hands you for contexts it scores.

Perturbations are split in two. Models with parameters fit them on the fit half; every model
is scored on the score half, so the reported numbers are comparable and none of them were
tuned on what they are scored against.

    python scripts/evaluate_local.py
    python scripts/evaluate_local.py --predict K562 --from RPE1   # the reverse fold

Read the numbers as relative signal. See src/vcc2026/local_eval.py for where they depart
from the official suite, and docs/validation_strategy.md for why RPE1 makes this optimistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize

from vcc2026.local_eval import (
    _own_gene_columns,
    expression_error,
    fold_change,
    load_profiles,
    log_profiles,
    means_from_fold_change,
    score,
)
from vcc2026.registry import ExperimentRecord, append_record, current_timestamp, default_run_id

CONTROL_LABEL = "control"

# (higher is better, no-information value) per metric
METRIC_SENSE = {
    "pds": (True, 0.5),
    "expr_error": (False, 1.0),
    "direction": (True, 0.5),
    "jaccard": (True, 0.0),
}


def _make_expression_objective(
    measured_means: np.ndarray,
    control_means: np.ndarray,
    perturbations: list[str],
    genes: list[str],
):
    """Expression error of a candidate log fold change, in the space the metric measures.

    Weights cannot be solved in closed form: they are applied to a log fold change, but the
    error is read off log-normalized profiles, and the two are not related linearly. A direct
    search over the metric is both simpler to justify and exactly what we report.
    """

    own_columns = _own_gene_columns(perturbations, genes)
    control_log = log_profiles(control_means.reshape(1, -1))
    measured_effect = log_profiles(measured_means) - control_log

    def objective(log_ratio: np.ndarray) -> float:
        means = means_from_fold_change(log_ratio, control_means)
        predicted_effect = log_profiles(means) - control_log
        return expression_error(predicted_effect, measured_effect, own_columns=own_columns)

    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", default="external/profiles")
    parser.add_argument("--predict", default="RPE1", help="cell line to be scored")
    parser.add_argument("--from", dest="source", default="K562", help="cell line to fit on")
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--registry", default="experiments/registry.jsonl")
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args()

    root = Path(args.profiles_dir)
    target = load_profiles(root / f"{args.predict}_perturbation_means.npz")
    source = load_profiles(root / f"{args.source}_perturbation_means.npz")

    # Align both lines onto the genes and perturbations they share.
    genes = [g for g in target.genes if g in set(source.genes)]
    target_gene_cols = [target.genes.index(g) for g in genes]
    source_gene_cols = [source.genes.index(g) for g in genes]

    shared_perts = set(source.perturbations) & set(target.perturbations)
    usable = [
        p
        for p in target.perturbations
        if p in shared_perts
        and p != CONTROL_LABEL
        and target.counts[target.index_of(p)] >= args.min_cells
    ]

    order = np.random.default_rng(args.seed).permutation(len(usable))
    n_fit = int(len(usable) * args.fit_fraction)
    fit_rows, score_rows = order[:n_fit], order[n_fit:]
    scored = [usable[i] for i in score_rows]

    target_control = target.means[target.index_of(CONTROL_LABEL), target_gene_cols]
    source_control = source.means[source.index_of(CONTROL_LABEL), source_gene_cols]
    measured_means = target.means[
        np.ix_([target.index_of(p) for p in usable], target_gene_cols)
    ]
    source_means = source.means[np.ix_([source.index_of(p) for p in usable], source_gene_cols)]

    # Everything is built from the source line's log fold change, carried onto the target's
    # own control profile. Models differ only in how they reweight it.
    source_lfc = fold_change(source_means, source_control)

    # The shared response is what every perturbation does in common; the remainder is what
    # distinguishes one from another. The two are orthogonal, so each weight solves
    # independently. Both are fit on the fit half only -- in the Challenge these constants
    # would come from public cell-line pairs rather than from the scored context.
    shared = source_lfc[fit_rows].mean(axis=0)
    specific = source_lfc - shared

    objective = _make_expression_objective(
        measured_means[fit_rows],
        target_control,
        [usable[i] for i in fit_rows],
        genes,
    )
    uniform = float(
        optimize.minimize_scalar(
            lambda s: objective(source_lfc[fit_rows] * s), bounds=(0.0, 2.0), method="bounded"
        ).x
    )
    beta, alpha = optimize.minimize(
        lambda w: objective(shared * w[0] + specific[fit_rows] * w[1]),
        x0=np.array([1.0, uniform]),
        method="Nelder-Mead",
    ).x

    measured = measured_means[score_rows]
    models = {
        "control": np.broadcast_to(target_control, measured.shape),
        "mean_response": np.broadcast_to(measured.mean(axis=0), measured.shape),
        "transfer": means_from_fold_change(source_lfc[score_rows], target_control),
        # Scale is applied before nonlinear log normalization; PDS can change as well.
        "scaled": means_from_fold_change(source_lfc[score_rows] * uniform, target_control),
        "calibrated": means_from_fold_change(
            shared * beta + specific[score_rows] * alpha, target_control
        ),
    }

    print(f"predicting {args.predict} from {args.source}")
    print(f"  {len(usable):,} perturbations usable, split {len(fit_rows):,} fit / {len(scored):,} scored")
    print(f"  {len(genes):,} shared genes")
    print(f"  uniform scale x {uniform:.3f}")
    print(f"  calibration: shared x {beta:.3f}, perturbation-specific x {alpha:.3f}\n")

    results = {
        name: score(
            np.ascontiguousarray(prediction),
            measured,
            target_control,
            perturbations=scored,
            genes=genes,
            top_n=args.top_n,
        )
        for name, prediction in models.items()
    }

    metrics = list(METRIC_SENSE)
    header = f"{'model':<16}" + "".join(f"{m:>13}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, values in results.items():
        print(f"{name:<16}" + "".join(f"{values[m]:>13.4f}" for m in metrics))
    print()
    for metric in metrics:
        higher, null = METRIC_SENSE[metric]
        print(
            f"  {metric:<12} {'higher' if higher else 'lower':>6} is better,"
            f" no information = {null}"
        )

    if not args.no_registry:
        record = ExperimentRecord(
            run_id=default_run_id("localeval"),
            timestamp_utc=current_timestamp(),
            git_commit="unknown",
            dataset_version=f"replogle-{args.source}-to-{args.predict}",
            seed=args.seed,
            model_name="local_eval_reference_models",
            config={
                "predict": args.predict,
                "source": args.source,
                "min_cells": args.min_cells,
                "top_n": args.top_n,
                "fit_fraction": args.fit_fraction,
                "n_fit": len(fit_rows),
                "n_scored": len(scored),
                "n_genes": len(genes),
                "beta_shared": beta,
                "alpha_specific": alpha,
            },
            metrics={f"{name}.{m}": v for name, vals in results.items() for m, v in vals.items()},
            training_time_seconds=None,
            compute="cpu",
        )
        append_record(args.registry, record)
        print(f"\nlogged to {args.registry} as {record.run_id}")

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
