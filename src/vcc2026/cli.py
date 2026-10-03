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
from .transfer import (
    apply_neighbor_effects,
    blend_fold_changes,
    build_transfer_prediction,
    neighbor_fold_changes,
    secondary_fold_changes,
    source_fold_changes,
)


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
        "--cell-construction", choices=["floorceil", "conditioned_nb"], default="floorceil",
        help="floorceil splits a mean between its floor and ceiling, which is the tightest "
        "an integer group can be and produces thousands of spurious DE calls against real "
        "control cells. conditioned_nb reproduces the mean exactly while spreading the cells "
        "like real ones, and needs the profile at full depth to stay clean.",
    )
    transfer.add_argument(
        "--entries-per-cell", type=float, default=None,
        help="Density budget. Total stored entries are this x 360,000; vcc prep needs "
        "roughly 12 bytes per entry to package.",
    )
    transfer.add_argument(
        "--secondary", default=None,
        help="A second screen's per-target effects, as written by scripts/fetch_cd4_effects.py. "
        "Averaged into the primary source to denoise each target's own response.",
    )
    transfer.add_argument(
        "--secondary-weight", type=float, default=0.10,
        help="How much of the blend the second source gets. Held out in both directions "
        "between Replogle's lines, 0.10 improves all four local metrics and 0.50 is worse "
        "than the primary alone on discrimination -- the second source is the weaker "
        "measurement, so it is a correction, not an equal partner.",
    )
    transfer.add_argument(
        "--neighbor-embeddings", default=None,
        help="scGPT gene embeddings from scripts/extract_scgpt_embeddings.py. Turns on FAISS "
        "retrieval of each target's nearest genes among the source screen's perturbations; "
        "their effects fill targets no screen measured and, with --neighbor-weight, are mixed "
        "into measured ones. Off by default. See docs/findings.md section 15 for how much "
        "signal there is.",
    )
    transfer.add_argument("--neighbor-k", type=int, default=20,
                          help="Neighbours retrieved per target.")
    transfer.add_argument(
        "--neighbor-temperature", type=float, default=None,
        help="Softmax temperature over neighbour similarity. Omit for a plain mean.",
    )
    transfer.add_argument(
        "--neighbor-weight", type=float, default=0.0,
        help="Share of the neighbours' effect mixed into targets that were measured: "
        "(1 - w) * own + w * neighbours.",
    )
    transfer.add_argument(
        "--neighbor-fill", action=argparse.BooleanOptionalAction, default=True,
        help="Give targets no screen measured their neighbours' effect rather than the "
        "generic response. Only acts when --neighbor-embeddings is set.",
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

    blend_scale = None
    if args.secondary:
        second, second_covered, measured = secondary_fold_changes(
            args.secondary, targets, gene_names
        )
        print(f"{second_covered.sum()}/{len(targets)} covered by "
              f"{Path(args.secondary).name} over {measured.sum():,} genes"
              f" ({(second_covered & ~covered).sum()} of them new)")
        fold_changes, covered, blend_scale = blend_fold_changes(
            fold_changes, covered, second, second_covered, measured,
            weight=args.secondary_weight,
        )
        print(f"blended at weight {args.secondary_weight} after scaling the second"
              f" source by {blend_scale:.3f} to match effect size")

    neighbor_filled = neighbor_blended = None
    if args.neighbor_embeddings:
        borrowed, retrieved, borrowed_measured = neighbor_fold_changes(
            args.source, args.neighbor_embeddings, targets, gene_names,
            k=args.neighbor_k, temperature=args.neighbor_temperature,
        )
        fold_changes, covered, neighbor_filled, neighbor_blended = apply_neighbor_effects(
            fold_changes, covered, borrowed, retrieved, borrowed_measured,
            weight=args.neighbor_weight, fill=args.neighbor_fill,
        )
        print(f"{retrieved.sum()}/{len(targets)} targets have a scGPT embedding: "
              f"{neighbor_filled} filled from {args.neighbor_k} neighbours, "
              f"{neighbor_blended} blended at weight {args.neighbor_weight}")

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
            with_dispersion=args.cell_construction == "conditioned_nb",
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
        construction=args.cell_construction,
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
            "cell_construction": args.cell_construction,
            "targets_covered": int(covered.sum()),
            "secondary": args.secondary,
            "secondary_weight": args.secondary_weight if args.secondary else None,
            "secondary_scale": blend_scale,
            "neighbor_embeddings": args.neighbor_embeddings,
            "neighbor_k": args.neighbor_k if args.neighbor_embeddings else None,
            "neighbor_temperature": args.neighbor_temperature if args.neighbor_embeddings else None,
            "neighbor_weight": args.neighbor_weight if args.neighbor_embeddings else None,
            "neighbor_filled": neighbor_filled,
            "neighbor_blended": neighbor_blended,
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
