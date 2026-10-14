"""Run scGPT over a context's control cells and cache what it makes of each gene there.

For each context this keeps, over disjoint halves of its cells: every cell's <cls> embedding,
and each gene's transformer hidden state summed over the cells that expressed it. The halves
exist so that "how different is a gene between two contexts" can be compared with "how
different is it between two samples of the same context".

    python scripts/embed_contexts.py --context K562 --cells 300
    python scripts/embed_contexts.py --context A --cells 300

K562 and RPE1 control cells are read by row from figshare's single-cell files over HTTP range
requests (the files are 9-11 GB; ~35 KB per cell is read). A, B, C come from dataset/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.cell_validation import read_rows
from vcc2026.io import _index_names, read_name_list
from vcc2026.remote import HttpRangeFile
from vcc2026.scgpt_model import encode_cells, load_scgpt

REMOTE = {
    "K562": "https://ndownloader.figshare.com/files/35773219",
    "RPE1": "https://ndownloader.figshare.com/files/35775606",
}
CONTROL_LABEL = "non-targeting"


def _fetch_rows(url: str, rows: list[int]) -> np.ndarray:
    """One worker's share. Each request costs seconds of latency, not bandwidth, and h5py
    serialises threads behind a global lock, so the reads are spread over processes."""
    with h5py.File(HttpRangeFile(url, block=1 << 17, cached_blocks=64), "r") as handle:
        x = handle["X"]
        return np.stack([x[int(r)] for r in rows]).astype(np.float32)


def remote_controls(url: str, n: int, seed: int, workers: int = 24) -> tuple[np.ndarray, list[str]]:
    from concurrent.futures import ProcessPoolExecutor

    with h5py.File(HttpRangeFile(url), "r") as handle:
        codes = handle["obs/gene"][()]
        labels = [c.decode() if isinstance(c, bytes) else str(c) for c in handle["obs/__categories/gene"][()]]
        rows = np.flatnonzero(codes == labels.index(CONTROL_LABEL))
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=n, replace=False))
        # The index is Ensembl IDs; scGPT's vocabulary is gene symbols, kept as a categorical.
        names = [c.decode() if isinstance(c, bytes) else str(c) for c in handle["var/__categories/gene_name"][()]]
        symbols = [names[c] for c in handle["var/gene_name"][()]]
    shares = [rows[i::workers].tolist() for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        blocks = list(pool.map(_fetch_rows, [url] * workers, shares))
    counts = np.empty((n, blocks[0].shape[1]), dtype=np.float32)
    for i, block in enumerate(blocks):
        counts[i::workers] = block
    return counts, symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, choices=["K562", "RPE1", "A", "B", "C"])
    parser.add_argument("--cells", type=int, default=300, help="cells per half")
    parser.add_argument("--halves", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default="external/scgpt")
    parser.add_argument("--out-dir", default="external/profiles/scgpt_context")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    total = args.cells * args.halves
    started = time.perf_counter()
    if args.context in REMOTE:
        counts, symbols = remote_controls(REMOTE[args.context], total, args.seed)
    else:
        path = f"dataset/context_{args.context}.h5ad"
        rows = np.sort(np.random.default_rng(args.seed).choice(18400, size=total, replace=False))
        counts, symbols = read_rows(path, rows), read_name_list("dataset/gene_names.csv")
    print(f"{args.context}: read {counts.shape[0]} control cells x {counts.shape[1]:,} genes "
          f"in {time.perf_counter() - started:.0f}s", flush=True)

    model, vocab = load_scgpt(args.checkpoint_dir)
    out = {"symbols": np.asarray(symbols)}
    for half in range(args.halves):
        block = counts[half * args.cells : (half + 1) * args.cells]
        result = encode_cells(model, vocab, block, symbols, seed=args.seed + half, progress=True)
        out[f"cell_{half}"] = result["cell"]
        out[f"gene_sum_{half}"] = result["gene_sum"].astype(np.float32)
        out[f"gene_count_{half}"] = result["gene_count"].astype(np.float32)
        out["genes"] = np.asarray(result["genes"])
        # Plain expression statistics on the same cells, for controlling comparisons later.
        out["mean_counts"] = counts.mean(0)
        print(f"  half {half}: done at {time.perf_counter() - started:.0f}s", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    target = Path(args.out_dir) / f"{args.context}.npz"
    np.savez_compressed(target, **out)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
