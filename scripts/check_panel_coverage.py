"""Does a public dataset perturb the genes the Challenge asks about?

A transfer model needs the Challenge's target genes to have been knocked down somewhere.
Replogle's essential panels cover none of the 300, because they select common-essential
genes while the Challenge selected for a spread of effect sizes. This checks any h5ad for
that coverage, reading only the category labels rather than the expression matrix.

    python scripts/check_panel_coverage.py external/ReplogleWeissman2022_K562_gwps.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from vcc2026.io import read_csv_column, read_name_list


def _decode(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--perts", default="dataset/pert_counts.csv")
    parser.add_argument("--genes", default="dataset/gene_names.csv")
    parser.add_argument("--pert-col", default="perturbation")
    parser.add_argument("--min-cells", type=int, default=25)
    args = parser.parse_args()

    targets = read_csv_column(args.perts, expected_header="target_gene")
    measured = read_name_list(args.genes)
    print(f"challenge panel: {len(targets)} targets, {len(measured):,} measured genes\n")

    for name in args.files:
        path = Path(name)
        if not path.exists():
            print(f"=== {path.name} === not downloaded yet\n")
            continue

        with h5py.File(path, "r") as handle:
            matrix = handle["X"]
            dense = isinstance(matrix, h5py.Dataset)
            shape = matrix.shape if dense else tuple(int(v) for v in matrix.attrs["shape"])

            node = handle[f"obs/{args.pert_col}"]
            categories = np.array(_decode(node["categories"][()]))
            counts = np.bincount(node["codes"][()], minlength=len(categories))
            genes = set(_decode(handle["var"][handle["var"].attrs.get("_index", "_index")][()]))

        enough = {c for c, n in zip(categories, counts) if n >= args.min_cells}
        covered = [t for t in targets if t in enough]
        missing = [t for t in targets if t not in enough]

        print(f"=== {path.name} ===")
        print(f"  {shape[0]:,} cells x {shape[1]:,} genes, X {'dense' if dense else 'sparse'}")
        print(f"  {len(categories):,} perturbation labels, "
              f"{len(enough):,} with >= {args.min_cells} cells")
        print(f"  challenge targets covered : {len(covered):3d}/{len(targets)} "
              f"({len(covered) / len(targets):.1%})")
        print(f"  challenge genes measured  : {len(genes & set(measured)):,}/{len(measured):,}")
        if covered:
            cells = [int(counts[list(categories).index(t)]) for t in covered]
            print(f"  cells per covered target  : median {int(np.median(cells))}, "
                  f"min {min(cells)}, max {max(cells)}")
        if missing:
            print(f"  missing ({len(missing)}): {', '.join(missing[:10])}"
                  f"{' ...' if len(missing) > 10 else ''}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
