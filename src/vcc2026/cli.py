from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import fit_context_mean_baseline, generic_perturbation_response
from .challenge import format_manifest, load_manifest
from .io import load_control_gene_names, read_csv_column, read_name_list, validate_gene_alignment
from .registry import ExperimentRecord, append_record, current_timestamp, default_run_id
from .submission import write_perturbation_h5ad, write_prediction_h5ad
from .transfer import build_transfer_prediction, source_fold_changes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vcc2026")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Print challenge manifest and data summary.")
    inspect.add_argument("--manifest", default="dataset/manifest.json")
    inspect.add_argument("--genes", default="dataset/gene_names.csv")
    inspect.add_argument("--perts", default="dataset/pert_counts.csv")
    inspect.add_argument("--controls-dir", default="dataset")

    baseline = sub.add_parser("build-baseline", help="Build the context-mean baseline prediction.")
    baseline.add_argument("--controls-dir", default="dataset")
    baseline.add_argument("--genes", default="dataset/gene_names.csv")
    baseline.add_argument("--perts", default="dataset/pert_counts.csv")
    baseline.add_argument("--output", default="artifacts/baseline_prediction.h5ad")
    baseline.add_argument("--registry", default="experiments/registry.jsonl")
    baseline.add_argument("--run-id", default=None)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--cells-per-pert", type=int, default=400)
    baseline.add_argument(
        "--response-profiles",
        nargs="*",
        default=[
            "external/profiles/K562_perturbation_means.npz",
            "external/profiles/RPE1_perturbation_means.npz",
        ],
        help="Public profiles to take the generic perturbation direction from. "
        "Pass none to predict the control unchanged.",
    )
    baseline.add_argument("--response-scale", type=float, default=1.0)

    transfer = sub.add_parser(
        "build-transfer", help="Predict each perturbation from its measured effect in gwps."
    )
    transfer.add_argument("--controls-dir", default="dataset")
    transfer.add_argument("--genes", default="dataset/gene_names.csv")
    transfer.add_argument("--perts", default="dataset/pert_counts.csv")
    transfer.add_argument("--source", default="external/profiles/K562gw_perturbation_means.npz")
    transfer.add_argument("--output", default="artifacts/transfer_prediction.h5ad")
    transfer.add_argument("--registry", default="experiments/registry.jsonl")
    transfer.add_argument("--run-id", default=None)
    transfer.add_argument("--seed", type=int, default=0)
    transfer.add_argument("--cells-per-pert", type=int, default=400)
    transfer.add_argument("--cell-patterns", type=int, default=400)
    transfer.add_argument("--coverage", choices=["all_expressed", "legacy_budget"],
                          default="all_expressed")
    transfer.add_argument(
        "--profile-scale", type=float, default=1.0,
        help="Count-depth multiplier. Changes integerized cell distributions and DE tests.",
    )
    transfer.add_argument("--scale", type=float, default=0.3)
    transfer.add_argument(
        "--entries-per-cell", type=float, default=None,
        help="Density budget. Total stored entries are this x 360,000; vcc prep needs "
        "roughly 12 bytes per entry to package.",
    )
    transfer.add_argument(
        "--fallback-profiles",
        nargs="*",
        default=[
            "external/profiles/K562_perturbation_means.npz",
            "external/profiles/RPE1_perturbation_means.npz",
        ],
        help="Used for targets the source screen did not perturb.",
    )

    return parser


def cmd_build_transfer(args: argparse.Namespace) -> int:
    manifest = load_manifest("dataset/manifest.json")
    gene_names = read_name_list(args.genes)
    targets = read_csv_column(args.perts, expected_header="target_gene")

    fold_changes, covered = source_fold_changes(args.source, targets, gene_names)
    print(f"{covered.sum()}/{len(targets)} targets covered by {Path(args.source).name}")

    available = [p for p in (args.fallback_profiles or []) if Path(p).exists()]
    fallback = generic_perturbation_response(available, gene_names) if available else None
    if fallback is not None and (~covered).any():
        print(f"{(~covered).sum()} uncovered targets fall back to the generic response")

    predictions = {}
    for ctx in manifest.contexts:
        control_path = Path(args.controls_dir) / f"context_{ctx}.h5ad"
        mismatch = validate_gene_alignment(gene_names, load_control_gene_names(control_path))
        if mismatch:
            raise ValueError(f"{control_path}: {'; '.join(mismatch)}")
        predictions[ctx] = build_transfer_prediction(
            control_path,
            targets=targets,
            gene_names=gene_names,
            target_fold_changes=fold_changes,
            fallback=fallback,
            scale=args.scale,
            entries_per_cell=args.entries_per_cell,
            profile_scale=args.profile_scale,
            cell_patterns=args.cell_patterns,
            coverage=args.coverage,
        )
    first = predictions[manifest.contexts[0]]
    cost = np.minimum(first.means[0], 1.0).sum()
    print(
        f"carrying {first.gene_indices.size:,} genes at profile scale {args.profile_scale}"
        f" -> {cost:,.0f} entries/cell, {cost * 360_000 / 1e9:.2f}e9 total"
    )

    output = write_perturbation_h5ad(
        args.output,
        gene_names=gene_names,
        perturbations=targets,
        predictions=predictions,
        cells_per_pert=args.cells_per_pert,
        cell_patterns=args.cell_patterns,
        seed=args.seed,
    )

    record = ExperimentRecord(
        run_id=args.run_id or default_run_id("transfer"),
        timestamp_utc=current_timestamp(),
        git_commit="unknown",
        dataset_version=manifest.panel_id,
        seed=args.seed,
        model_name="gwps_transfer",
        config={
            "source": args.source,
            "scale": args.scale,
            "entries_per_cell": args.entries_per_cell,
            "profile_scale": args.profile_scale,
            "genes_carried": int(first.gene_indices.size),
            "cells_per_pert": args.cells_per_pert,
            "cell_patterns": args.cell_patterns,
            "coverage": args.coverage,
            "targets_covered": int(covered.sum()),
        },
        metrics={},
        training_time_seconds=None,
        compute="cpu",
    )
    append_record(args.registry, record)
    print(json.dumps({"output": str(output), "run_id": record.run_id}, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    print(format_manifest(manifest))
    expected_genes = read_name_list(args.genes)
    print(f"gene_names.csv count: {len(expected_genes)}")
    print(f"perturbation count: {len(read_csv_column(args.perts, expected_header='target_gene'))}")
    for ctx in manifest.contexts:
        path = Path(args.controls_dir) / f"context_{ctx}.h5ad"
        observed_genes = load_control_gene_names(path)
        mismatch = validate_gene_alignment(expected_genes, observed_genes)
        print(f"{ctx}: {path}")
        print(f"  genes_aligned: {not mismatch}")
        if mismatch:
            print(f"  mismatch: {mismatch[0]}")
    return 0


def cmd_build_baseline(args: argparse.Namespace) -> int:
    manifest = load_manifest("dataset/manifest.json")
    gene_names = read_name_list(args.genes)
    perts = read_csv_column(args.perts, expected_header="target_gene")

    available = [p for p in (args.response_profiles or []) if Path(p).exists()]
    shift = generic_perturbation_response(available, gene_names) if available else None
    if shift is not None:
        covered = int(np.count_nonzero(shift))
        print(f"generic response from {len(available)} profile(s): {covered:,} genes carry a direction")
    elif args.response_profiles:
        print("no response profiles found; predicting the control unchanged")

    baselines = {}
    for ctx in manifest.contexts:
        control_path = Path(args.controls_dir) / f"context_{ctx}.h5ad"
        control_gene_names = load_control_gene_names(control_path)
        mismatch = validate_gene_alignment(gene_names, control_gene_names)
        if mismatch:
            raise ValueError(f"{control_path}: {'; '.join(mismatch)}")
        baselines[ctx] = fit_context_mean_baseline(
            control_path, response_shift=shift, response_scale=args.response_scale
        )

    output = write_prediction_h5ad(
        args.output,
        gene_names=gene_names,
        perturbations=perts,
        baselines=baselines,
        cells_per_pert=args.cells_per_pert,
    )

    record = ExperimentRecord(
        run_id=args.run_id or default_run_id("baseline"),
        timestamp_utc=current_timestamp(),
        git_commit="unknown",
        dataset_version=manifest.panel_id,
        seed=args.seed,
        model_name="context_mean_baseline",
        config={
            "controls_dir": args.controls_dir,
            "genes": args.genes,
            "perts": args.perts,
            "output": args.output,
            "cells_per_pert": args.cells_per_pert,
        },
        metrics={},
        training_time_seconds=None,
        compute="cpu",
    )
    append_record(args.registry, record)

    print(json.dumps({"output": str(output), "run_id": record.run_id, "registry": args.registry}, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "build-baseline":
        return cmd_build_baseline(args)
    if args.command == "build-transfer":
        return cmd_build_transfer(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
