"""Reduce the external Replogle files to per-perturbation mean profiles.

The raw files are ~2.8 GB on disk and expand to ~9-11 GB each if read whole, which makes
them impractical to touch repeatedly. Averaging cells within each perturbation collapses
them to a few tens of MB that load instantly, which is the form every downstream comparison
needs anyway.

    python scripts/build_external_profiles.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from vcc2026.io import group_mean_profiles, read_var_names

FILES = {
    "K562": "ReplogleWeissman2022_K562_essential.h5ad",
    "RPE1": "ReplogleWeissman2022_rpe1.h5ad",
    "K562gw": "ReplogleWeissman2022_K562_gwps.h5ad",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", default="external")
    parser.add_argument("--out-dir", default="external/profiles")
    parser.add_argument("--pert-col", default="perturbation")
    parser.add_argument("--row-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--lines", nargs="*", choices=list(FILES), default=["K562", "RPE1"],
        help="Which cell lines to reduce. K562gw is the genome-wide screen: 2M cells, "
        "66 GB to stream, and the only source covering the Challenge's target panel.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for line in args.lines:
        filename = FILES[line]
        path = Path(args.external_dir) / filename
        print(f"{line}: reading {path}")
        started = time.perf_counter()

        genes = read_var_names(path)
        means, perturbations, counts = group_mean_profiles(
            path, args.pert_col, row_chunk_size=args.row_chunk_size
        )

        output = out_dir / f"{line}_{args.pert_col}_means.npz"
        np.savez_compressed(
            output,
            means=means,
            perturbations=np.asarray(perturbations),
            counts=counts,
            genes=np.asarray(genes),
        )

        elapsed = time.perf_counter() - started
        size_mb = output.stat().st_size / 1e6
        print(
            f"{line}: {means.shape[0]:,} perturbations x {means.shape[1]:,} genes"
            f" | {counts.sum():,} cells | median {int(np.median(counts))} cells/pert"
            f" -> {output} ({size_mb:.1f} MB, {elapsed / 60:.1f} min)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
