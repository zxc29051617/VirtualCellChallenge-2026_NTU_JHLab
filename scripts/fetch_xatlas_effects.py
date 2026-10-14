"""Pull X-Atlas/Orion (HCT116) per-target effects for the Challenge's targets, without 2.4 GB.

Source: Arc's PIE preprocessed set, a differential-expression summary of Huang et al. 2025
(genome-wide CRISPRi Perturb-seq in HCT116 and HEK293T; CC BY-NC-SA 4.0). `lfc_true.npy` is
(13,284 rows, 22,182 genes) float64 of log2 fold changes, NaN where a gene was not tested. It is
a plain row-major .npy, so one row is 177 kB of adjacent bytes and a range request fetches it.

Only perturbations that passed the PIE filters are in it: 146 of the Challenge's 300 targets in
HCT116, and 11 in HEK293T. The raw per-cell files (209 and 350 GB) hold all 300 but cost hours
to read, since their rows are not grouped by target.

    python scripts/fetch_xatlas_effects.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/datasets/arcinstitute/PIE_x_atlas_orion/resolve/main/preprocessed"


def _range(url: str, start: int, stop: int, attempts: int = 6) -> bytes:
    import time

    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{stop}", "User-Agent": "Mozilla/5.0"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def npy_layout(url: str) -> tuple[tuple[int, ...], np.dtype, int]:
    """(shape, dtype, data offset) of a remote .npy, from its header alone."""
    import io

    head = _range(url, 0, 4095)
    stream = io.BytesIO(head)
    version = np.lib.format.read_magic(stream)
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    if fortran:
        raise ValueError("expected a C-ordered array")
    return shape, np.dtype(dtype), stream.tell()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="hct116", choices=["hct116", "hek293t"])
    parser.add_argument("--perts", default="dataset/pert_counts.csv")
    parser.add_argument("--meta-dir", default="external/xatlas")
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-tested", type=float, default=0.9,
                        help="keep genes tested for at least this share of targets")
    args = parser.parse_args()

    meta = json.load(open(Path(args.meta_dir) / "meta.json"))
    perts = np.load(Path(args.meta_dir) / "pert_ids.npy")
    ctx = np.load(Path(args.meta_dir) / "ctx_ids.npy")
    context_id = meta["context_to_id"][args.context]
    pert_id = meta["pert_to_id"]
    genes = list(meta["genes"])

    targets = [l.strip() for l in Path(args.perts).read_text().splitlines()[1:] if l.strip()]
    inverse = {v: k for k, v in pert_id.items()}
    row_of = {inverse[int(perts[row])]: int(row) for row in np.flatnonzero(ctx == context_id)}
    wanted = [t for t in targets if t in row_of]
    print(f"{args.context}: {len(wanted)}/{len(targets)} Challenge targets are in the preprocessed set")

    url = f"{BASE}/lfc_true.npy"
    shape, dtype, offset = npy_layout(url)
    row_bytes = shape[1] * dtype.itemsize
    print(f"lfc_true.npy {shape} {dtype}, row = {row_bytes / 1e3:.0f} kB, data starts at byte {offset}")

    def fetch(target: str) -> np.ndarray:
        start = offset + row_of[target] * row_bytes
        return np.frombuffer(_range(url, start, start + row_bytes - 1), dtype=dtype).copy()

    with ThreadPoolExecutor(12) as pool:
        rows = list(pool.map(fetch, wanted))
    lfc = np.stack(rows)

    tested = (~np.isnan(lfc)).mean(0)
    keep = tested >= args.min_tested
    print(f"{int(keep.sum()):,} of {len(genes):,} genes tested for >= {args.min_tested:.0%} of targets")
    effects = np.nan_to_num(lfc[:, keep] * np.log(2)).astype(np.float32)  # natural log

    output = Path(args.output or f"external/profiles/{args.context.upper()}_target_lfc.npz")
    np.savez_compressed(
        output,
        log_fc=effects,
        targets=np.asarray(wanted),
        conditions=np.asarray([args.context] * len(wanted)),
        genes=np.asarray([g for g, k in zip(genes, keep) if k]),
    )
    print(f"wrote {output}: {effects.shape[0]} targets x {effects.shape[1]:,} genes, "
          f"median |log fc| {np.median(np.abs(effects)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
