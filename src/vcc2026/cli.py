from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import fit_context_mean_baseline, generic_perturbation_response
from .challenge import format_manifest, load_manifest
from .io import load_control_gene_names, read_csv_column, read_name_list, validate_gene_alignment
from .registry import ExperimentRecord, append_record, current_timestamp, default_run_id
from .submission import write_prediction_h5ad


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

    return parser


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
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
