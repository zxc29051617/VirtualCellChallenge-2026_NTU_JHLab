"""Score models on a held-out cell line, as a local stand-in for the Challenge.

Fit on K562, predict RPE1. RPE1's perturbation profiles are the answer key and are never
shown to a model -- only its control profile and the list of genes to knock down, which is
exactly what the Challenge hands you for contexts it scores.

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

from vcc2026.local_eval import CPM_TOTAL, LFC_EPSILON, load_profiles, score
from vcc2026.registry import ExperimentRecord, append_record, current_timestamp, default_run_id

CONTROL_LABEL = "control"

# (higher is better, no-information value) per metric
METRIC_SENSE = {
    "pds": (True, 0.5),
    "expr_error": (False, 1.0),
    "direction": (True, 0.5),
    "jaccard": (True, 0.0),
}


def _cpm(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=-1, keepdims=True)
    return matrix * (CPM_TOTAL / np.where(totals > 0, totals, 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", default="external/profiles")
    parser.add_argument("--predict", default="RPE1", help="cell line to be scored")
    parser.add_argument("--from", dest="source", default="K562", help="cell line to fit on")
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=100)
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

    shared = set(source.perturbations) & set(target.perturbations)
    scored = [
        p
        for p in target.perturbations
        if p in shared and p != CONTROL_LABEL and target.counts[target.index_of(p)] >= args.min_cells
    ]

    target_rows = [target.index_of(p) for p in scored]
    source_rows = [source.index_of(p) for p in scored]

    measured = target.means[np.ix_(target_rows, target_gene_cols)]
    target_control = target.means[target.index_of(CONTROL_LABEL), target_gene_cols]
    source_perturbed = source.means[np.ix_(source_rows, source_gene_cols)]
    source_control = source.means[source.index_of(CONTROL_LABEL), source_gene_cols]

    print(f"predicting {args.predict} from {args.source}")
    print(
        f"  {len(scored):,} perturbations scored "
        f"({len(shared):,} shared, dropped those under {args.min_cells} cells)"
    )
    print(f"  {len(genes):,} shared genes")
    print(f"  median {int(np.median(target.counts[target_rows]))} cells per perturbation\n")

    control_cpm = _cpm(target_control)
    models = {
        # Predict nothing changes.
        "control": np.broadcast_to(target_control, measured.shape),
        # The official baseline's shape: one averaged response, assigned to every
        # perturbation. It reads the answer key, so it is the yardstick, not a competitor.
        "mean_response": np.broadcast_to(measured.mean(axis=0), measured.shape),
        # Carry the fold change measured in the source line onto the target's own baseline.
        "transfer": control_cpm
        * ((_cpm(source_perturbed) + LFC_EPSILON) / (_cpm(source_control) + LFC_EPSILON)),
    }

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
        print(f"  {metric:<12} {'higher' if higher else 'lower':>6} is better,"
              f" no information = {null}")

    if not args.no_registry:
        record = ExperimentRecord(
            run_id=default_run_id("localeval"),
            timestamp_utc=current_timestamp(),
            git_commit="unknown",
            dataset_version=f"replogle-{args.source}-to-{args.predict}",
            seed=0,
            model_name="local_eval_reference_models",
            config={
                "predict": args.predict,
                "source": args.source,
                "min_cells": args.min_cells,
                "top_n": args.top_n,
                "n_perturbations": len(scored),
                "n_genes": len(genes),
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
