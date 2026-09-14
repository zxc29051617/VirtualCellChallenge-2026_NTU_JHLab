"""Print a plain-language tour of one challenge context file.

Intended as a first orientation step: it shows what a single row of the data
actually is, how sparse it is, and what shape the prediction has to be.

    python scripts/explore_data.py --context A
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from vcc2026.challenge import expected_prediction_cells, load_manifest
from vcc2026.io import read_csv_column, read_name_list


def read_rows(handle: h5py.File, start: int, stop: int, n_genes: int) -> np.ndarray:
    """Densify rows [start, stop) of a CSR matrix stored in an h5ad file."""
    indptr = handle["X/indptr"][start : stop + 1]
    lo, hi = int(indptr[0]), int(indptr[-1])
    indices = handle["X/indices"][lo:hi]
    data = handle["X/data"][lo:hi]
    dense = np.zeros((stop - start, n_genes), dtype=np.float32)
    for row in range(stop - start):
        a = int(indptr[row]) - lo
        b = int(indptr[row + 1]) - lo
        dense[row, indices[a:b]] = data[a:b]
    return dense


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="A")
    parser.add_argument("--controls-dir", default="dataset")
    parser.add_argument("--manifest", default="dataset/manifest.json")
    parser.add_argument("--genes", default="dataset/gene_names.csv")
    parser.add_argument("--perts", default="dataset/pert_counts.csv")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    gene_names = read_name_list(args.genes)
    perts = read_csv_column(args.perts, expected_header="target_gene")
    path = Path(args.controls_dir) / f"context_{args.context}.h5ad"

    with h5py.File(path, "r") as handle:
        n_cells, n_genes = (int(x) for x in handle["X"].attrs["shape"])
        total_nnz = int(handle["X/indptr"][-1])

        section("1. The file is one big matrix")
        print(f"file            : {path}")
        print(f"rows (cells)    : {n_cells:,}")
        print(f"columns (genes) : {n_genes:,}")
        print(f"cells x genes   : {n_cells * n_genes:,} numbers")
        print(f"non-zero        : {total_nnz:,} ({100 * total_nnz / (n_cells * n_genes):.1f}%)")
        print("Every value is an integer: how many mRNA molecules of that gene")
        print("were captured in that one cell. Most are 0, so it is stored sparse.")

        section("2. One cell = one row of counts")
        block = read_rows(handle, 0, 5, n_genes)
        expressed = np.flatnonzero(block[0] > 0)
        order = expressed[np.argsort(-block[0][expressed])][:8]
        print("Cell 0, its 8 most-expressed genes:")
        for gene_idx in order:
            print(f"  {gene_names[gene_idx]:<12} {int(block[0][gene_idx]):>6}")
        print()
        print("Library size (total counts) of the first 5 cells:")
        for row in range(5):
            n_expressed = int((block[row] > 0).sum())
            print(f"  cell {row}: {int(block[row].sum()):>7,} counts across {n_expressed:>6,} genes")
        print("Cells differ in depth, which is why raw counts cannot be compared directly.")

        section("3. These cells are all UNPERTURBED controls")
        for column in ["target_gene", "context", "ntc_id"]:
            group = handle[f"obs/{column}"]
            categories = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in group["categories"][()]
            ]
            shown = ", ".join(categories[:3])
            suffix = ", ..." if len(categories) > 3 else ""
            print(f"  obs['{column}']: {len(categories)} value(s) -> {shown}{suffix}")
        print("'non-targeting' means the CRISPRi guide targets nothing, so nothing")
        print("was silenced. This is the baseline state of the cell line.")

    section("4. What you must predict")
    measured = set(gene_names)
    hits = sum(1 for pert in perts if pert in measured)
    print(f"target genes to knock down : {len(perts)} (e.g. {', '.join(perts[:5])})")
    print(f"  of those, also measured  : {hits}/{len(perts)}")
    print(f"contexts                   : {', '.join(manifest.contexts)}")
    print(f"cells per perturbation     : {manifest.cells_per_pert}")
    print(
        f"prediction matrix          : {expected_prediction_cells(manifest):,} rows"
        f" x {manifest.n_genes:,} genes"
    )
    print()
    print("So: you get the unperturbed matrix above plus a list of gene names,")
    print("and must invent the post-knockdown matrix. You never see a single")
    print("perturbed cell from this cell line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
