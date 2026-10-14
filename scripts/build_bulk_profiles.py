"""Build per-perturbation mean profiles from figshare's pre-aggregated "raw_bulk" Replogle
files, instead of scPerturb's per-cell harmonized release.

The per-cell files (used by build_external_profiles.py) are 1.2-8.8 GB and slow to fetch on
a constrained connection. Figshare hosts the same underlying data already reduced to one row
per (gene, sgRNA) pseudobulk -- 80-375 MB -- which this script further collapses to one row
per target gene (weighted by cell count, combining sgRNA probes of the same target) and
writes in the exact .npz layout `local_eval.load_profiles` expects, so it drops in wherever
build_external_profiles.py's output was used.

    python scripts/build_bulk_profiles.py external/K562gwps_raw_bulk.h5ad K562gw
    python scripts/build_bulk_profiles.py external/K562_essential_raw_bulk.h5ad K562
    python scripts/build_bulk_profiles.py external/rpe1_raw_bulk.h5ad RPE1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_target(gene_transcript: str) -> str:
    parts = gene_transcript.split("_")
    target = parts[1] if len(parts) > 1 else gene_transcript
    # Match the rest of the pipeline's control label (transfer.py, evaluate_local.py default
    # to "control"), rather than this source's own "non-targeting" spelling.
    return "control" if target == "non-targeting" else target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to a *_raw_bulk.h5ad file")
    parser.add_argument("label", help="Short name for the output, e.g. K562gw")
    parser.add_argument("--out-dir", default="external/profiles")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        X = f["X"][()]  # dense, small enough (<= ~375 MB) to load whole
        index = np.array([v.decode() if isinstance(v, bytes) else str(v)
                           for v in f["obs/gene_transcript"][()]])
        cells = f["obs/num_cells_unfiltered"][()].astype(np.int64)

        gene_codes = f["var/gene_name"][()]
        categories = np.array([c.decode() if isinstance(c, bytes) else str(c)
                                for c in f["var/__categories/gene_name"][()]])
        genes = np.array([categories[c] if c >= 0 else "" for c in gene_codes])

    targets = np.array([parse_target(s) for s in index])

    order = np.argsort(targets, kind="stable")
    targets_sorted = targets[order]
    changed = targets_sorted[1:] != targets_sorted[:-1]
    run_starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
    unique_targets = targets_sorted[run_starts]

    n_out, n_genes = len(unique_targets), X.shape[1]
    means = np.zeros((n_out, n_genes), dtype=np.float32)
    counts = np.zeros(n_out, dtype=np.int64)

    run_ends = np.concatenate((run_starts[1:], [len(targets_sorted)]))
    for out_row, (start, end) in enumerate(zip(run_starts, run_ends)):
        rows = order[start:end]
        w = cells[rows].astype(np.float64)
        total_w = w.sum()
        if total_w <= 0:
            continue
        means[out_row] = (X[rows] * w[:, None]).sum(axis=0) / total_w
        counts[out_row] = int(w.sum())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.label}_perturbation_means.npz"
    np.savez_compressed(
        output,
        means=means,
        perturbations=unique_targets,
        counts=counts,
        genes=genes,
    )
    n_control = int(counts[unique_targets == "non-targeting"].sum()) if "non-targeting" in unique_targets else 0
    print(
        f"{args.label}: {n_out:,} targets (from {len(index):,} probe rows) x {n_genes:,} genes"
        f" | control cells: {n_control:,} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
