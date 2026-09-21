"""Pull CD4 CRISPRi per-gene effects for a target list, without the 16.8 GB download.

Why a second source at all: the perturbation-specific component transfers between Replogle's
own two cell lines at r = 0.024 (findings section 7). A signal that weak still ranks well --
it is what carries discrimination -- but averaging two independent measurements of the same
knockdown should denoise it, and discrimination is 41% of the leader's total.

CD4 covers 291 of the Challenge's 300 targets against gwps's 267, in three culture
conditions, and none of it touches the density that vcc prep cannot package.

    python scripts/fetch_cd4_effects.py --targets dataset/pert_counts.csv
    python scripts/fetch_cd4_effects.py --targets external/profiles/K562gw_perturbation_means.npz

Quality flags come from the authors' own annotations; `--no-qc` keeps every row.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from vcc2026.io import read_csv_column
from vcc2026.remote import HttpRangeFile

URL = (
    "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/"
    "marson2025_data/GWCD4i.DE_stats.h5ad"
)


def _decode(values) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _column(handle: h5py.File, name: str) -> np.ndarray:
    node = handle[f"obs/{name}"]
    if isinstance(node, h5py.Group):
        return np.asarray(_decode(node["categories"][()]))[node["codes"][()]]
    return node[()]


def _read_targets(path: Path) -> list[str]:
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        return [str(p) for p in payload["perturbations"] if str(p) != "control"]
    return read_csv_column(path, expected_header="target_gene")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="dataset/pert_counts.csv")
    parser.add_argument("--output", default="external/profiles/CD4_target_lfc.npz")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--min-cells", type=int, default=100)
    parser.add_argument("--no-qc", action="store_true")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    wanted = set(_read_targets(Path(args.targets)))
    print(f"{len(wanted):,} targets requested")

    remote = HttpRangeFile(args.url)
    started = time.perf_counter()
    with h5py.File(remote, "r") as handle:
        names = _column(handle, "target_contrast_gene_name")
        conditions = _column(handle, "culture_condition")
        genes = _decode(handle["var/gene_name"][()])

        keep = np.isin(names, list(wanted))
        if not args.no_qc:
            # The authors' own flags. on-target significance is what says the guide worked;
            # the rest mark a measurement whose effect may belong to a neighbouring gene.
            keep &= _column(handle, "ontarget_significant").astype(bool)
            keep &= ~_column(handle, "low_target_gex").astype(bool)
            keep &= ~_column(handle, "distal_offtarget_flag").astype(bool)
            keep &= ~_column(handle, "neighboring_gene_KD").astype(bool)
            keep &= np.nan_to_num(_column(handle, "n_cells_target")) >= args.min_cells

        rows = np.flatnonzero(keep)
        print(f"{len(rows):,} rows pass"
              f"{'' if args.no_qc else ' the authors quality flags'}"
              f", {len(set(names[rows].tolist())):,} distinct targets")
        print(f"labels cost {remote.bytes_fetched / 1e6:.0f} MB\n")

        layer = handle["layers/log_fc"]
        effects = np.empty((len(rows), len(genes)), dtype=np.float32)
        for start in range(0, len(rows), args.batch):
            block = rows[start: start + args.batch]
            effects[start: start + len(block)] = layer[np.sort(block), :]
            print(f"  {min(start + args.batch, len(rows)):>5,}/{len(rows):,} rows"
                  f"  {remote.bytes_fetched / 1e9:.2f} GB"
                  f"  {time.perf_counter() - started:.0f}s", end="\r")

    print(f"\n\nfetched {remote.bytes_fetched / 1e9:.2f} GB in {remote.requests:,} requests"
          f" over {time.perf_counter() - started:.0f}s")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        log_fc=effects,
        targets=np.asarray(names[rows]),
        conditions=np.asarray(conditions[rows]),
        genes=np.asarray(genes),
    )
    print(f"wrote {output} ({output.stat().st_size / 1e6:.0f} MB)")
    print(f"  {effects.shape[0]:,} rows x {effects.shape[1]:,} genes,"
          f" median |log fc| {np.nanmedian(np.abs(effects)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
