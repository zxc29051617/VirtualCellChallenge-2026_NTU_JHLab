"""Generate a "predict the per-context control mean" baseline submission.

For each context (A, B, C):
  1. Stream the context's .h5ad file in chunks (backed='r') to compute the mean
     control ("non-targeting") count vector, without ever materializing the
     full .X matrix in memory.
  2. For every target gene in pert_counts.csv, emit CELLS_PER_PERT predicted
     cells whose expression is exactly that context's control mean vector.

Every predicted row for a given context is *byte-for-byte identical* (the
baseline ignores which gene is being predicted). Naively materializing that
as a real repeated CSR array would be tens of GB, and gzip can't shrink it
much because its 32KB DEFLATE window is smaller than a single row
(~64KB) of this ~87%-dense mean vector, so it can't see the repetition.
Instead, by default this script stores each context's mean vector on disk
exactly once and uses an HDF5 *virtual dataset* to map every predicted row's
X/data and X/indices onto that one copy -- this is not a shortcut or an
approximation, `ad.read_h5ad(...).X` transparently returns the exact same
values a fully materialized file would, HDF5 does the expansion on read.
Pass --materialize to instead physically write out every row (slow, large
file) if some downstream tool can't handle virtual datasets.

Usage:
    python generate_baseline_submission.py
    python generate_baseline_submission.py --genes-limit 5 --cells-per-pert 3   # smoke test
    python generate_baseline_submission.py --materialize                        # physical copy of every row
"""

import argparse
import os
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "submissions")
OUT_PATH = os.path.join(OUT_DIR, "baseline_control_mean.h5ad")

CONTEXTS = ["A", "B", "C"]
CONTEXT_FILES = {c: os.path.join(DATA_DIR, f"context_{c}.h5ad") for c in CONTEXTS}
PERT_COUNTS_PATH = os.path.join(DATA_DIR, "pert_counts.csv")

CONTROL_LABEL = "non-targeting"
DEFAULT_CELLS_PER_PERT = 400
DEFAULT_MIN_CPM = 1.0  # drop genes below this counts-per-million to satisfy the VCC scoring density cap
MEAN_READ_CHUNK = 4000  # rows per backed-read chunk while computing the control mean
WRITE_BATCH_GENES = 10  # genes per on-disk write batch (keeps each batch a few hundred MB)
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4


def compute_context_mean(h5ad_path, min_cpm=DEFAULT_MIN_CPM):
    """Return (var_names, nz_indices, nz_values) for the mean control-cell count
    vector of one context, streaming .X in chunks so it is never fully loaded.

    Genes below `min_cpm` counts-per-million are zeroed out. The raw mean is
    ~87% dense, which alone blows past the VCC scoring server's hard density
    cap (it holds the whole matrix uncompressed); low-CPM genes carry almost
    no expression mass, so dropping them is the fix `vcc prep` itself suggests.
    """
    adata = ad.read_h5ad(h5ad_path, backed="r")
    is_control = (adata.obs["target_gene"] == CONTROL_LABEL).to_numpy()
    n_control = int(is_control.sum())
    if n_control == 0:
        raise ValueError(f"No {CONTROL_LABEL!r} cells found in {h5ad_path}")

    total = np.zeros(adata.n_vars, dtype=np.float64)
    n_obs = adata.n_obs
    for start in range(0, n_obs, MEAN_READ_CHUNK):
        end = min(start + MEAN_READ_CHUNK, n_obs)
        mask = is_control[start:end]
        if not mask.any():
            continue
        block = adata.X[start:end]
        if not mask.all():
            block = block[mask]
        total += np.asarray(block.sum(axis=0)).ravel()

    var_names = adata.var_names.to_numpy(copy=True)
    adata.file.close()

    mean = total / n_control
    if min_cpm > 0:
        cpm = mean / mean.sum() * 1e6
        mean = np.where(cpm >= min_cpm, mean, 0.0)
    # VCC scoring requires raw integer counts; round (this also further trims
    # density, since sub-0.5 means round down to 0).
    mean = np.round(mean).astype(np.float32)
    nz_indices = np.flatnonzero(mean).astype(np.int32)
    nz_values = mean[nz_indices].astype(np.float32)
    return var_names, nz_indices, nz_values


def load_target_genes(genes_limit=None):
    genes = pd.read_csv(PERT_COUNTS_PATH)["target_gene"].tolist()
    if genes_limit is not None:
        genes = genes[:genes_limit]
    return genes


def build_obs(genes, cells_per_pert):
    """obs rows are ordered context-major, then gene, then replicate index --
    this MUST match the order X rows are written in below."""
    n_genes = len(genes)
    per_context = n_genes * cells_per_pert

    target_gene_col = np.tile(np.repeat(np.asarray(genes, dtype=object), cells_per_pert), len(CONTEXTS))
    context_col = np.repeat(np.asarray(CONTEXTS, dtype=object), per_context)

    index = []
    for ctx in CONTEXTS:
        for gene in genes:
            index.extend(f"{ctx}_pred_{gene}_{k:03d}" for k in range(cells_per_pert))

    obs = pd.DataFrame(
        {
            "target_gene": pd.Categorical(target_gene_col),
            "context": pd.Categorical(context_col),
        },
        index=pd.Index(index, name=None),
    )
    return obs


def write_placeholder(out_path, n_rows, var_names, obs):
    """Write a correctly-shaped but empty (all-zero-nnz) AnnData so anndata's own
    writer produces a valid obs/var/uns skeleton; the X group is replaced afterward."""
    var = pd.DataFrame(index=pd.Index(var_names, name=None))
    placeholder_X = sp.csr_matrix((n_rows, len(var_names)), dtype=np.float32)
    adata = ad.AnnData(X=placeholder_X, obs=obs, var=var)
    adata.write_h5ad(out_path)


def _row_nnz_sequence_and_indptr(genes, context_means, cells_per_pert):
    n_genes = len(genes)
    row_nnz_per_context = {c: len(context_means[c][1]) for c in CONTEXTS}
    row_nnz_sequence = np.concatenate(
        [np.full(n_genes * cells_per_pert, row_nnz_per_context[c], dtype=np.int64) for c in CONTEXTS]
    )
    indptr = np.concatenate(([0], np.cumsum(row_nnz_sequence))).astype(np.int64)
    return row_nnz_per_context, indptr


def stream_write_x_virtual(out_path, genes, context_means, cells_per_pert, n_vars):
    """Replace the placeholder X group with a CSR matrix whose 'data'/'indices'
    arrays are HDF5 virtual datasets: each context's mean vector is stored on
    disk exactly once, and every predicted row is a mapping onto that single
    copy. Reads (via h5py/anndata) transparently return the full expanded
    array -- nothing downstream needs to know the storage is virtual."""
    n_genes = len(genes)
    row_nnz_per_context, indptr = _row_nnz_sequence_and_indptr(genes, context_means, cells_per_pert)
    total_nnz = int(indptr[-1])
    n_rows = len(indptr) - 1

    with h5py.File(out_path, "r+") as f:
        del f["X"]
        x_group = f.create_group("X")
        x_group.attrs["encoding-type"] = "csr_matrix"
        x_group.attrs["encoding-version"] = "0.1.0"
        x_group.attrs["shape"] = np.array([n_rows, n_vars], dtype=np.int64)

        src_group = f["uns"].create_group("_baseline_mean_source")
        sources = {}
        for ctx in CONTEXTS:
            _, nz_idx, nz_val = context_means[ctx]
            idx_ds = src_group.create_dataset(f"{ctx}_indices", data=nz_idx, dtype=np.int32)
            val_ds = src_group.create_dataset(f"{ctx}_data", data=nz_val, dtype=np.float32)
            for ds in (idx_ds, val_ds):
                ds.attrs["encoding-type"] = "array"
                ds.attrs["encoding-version"] = "0.2.0"
            sources[ctx] = (h5py.VirtualSource(idx_ds), h5py.VirtualSource(val_ds))

        data_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.float32)
        indices_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.int32)

        n_rows_per_context = n_genes * cells_per_pert
        pos = 0
        row = 0
        for ctx in CONTEXTS:
            idx_source, val_source = sources[ctx]
            row_nnz = row_nnz_per_context[ctx]
            for _ in range(n_rows_per_context):
                data_layout[pos:pos + row_nnz] = val_source
                indices_layout[pos:pos + row_nnz] = idx_source
                pos += row_nnz
                row += 1

        x_group.create_virtual_dataset("data", data_layout)
        x_group.create_virtual_dataset("indices", indices_layout)
        x_group.create_dataset("indptr", data=indptr, dtype=np.int64)

    print(f"  virtual dataset: {n_rows} rows / {total_nnz} logical nonzeros, "
          f"physically stored once per context ({sum(row_nnz_per_context.values())} values total)")


def stream_write_x_materialized(out_path, genes, context_means, cells_per_pert, n_vars):
    """Replace the placeholder X group with a streamed, chunked, gzip-compressed
    CSR matrix built batch-by-batch so the full matrix is never held in RAM.
    This physically duplicates every row's data -- large and slow; prefer
    stream_write_x_virtual unless a downstream tool can't read virtual datasets."""
    n_genes = len(genes)
    row_nnz_per_context, indptr = _row_nnz_sequence_and_indptr(genes, context_means, cells_per_pert)
    total_nnz = int(indptr[-1])
    n_rows = len(indptr) - 1

    with h5py.File(out_path, "r+") as f:
        del f["X"]
        x_group = f.create_group("X")
        x_group.attrs["encoding-type"] = "csr_matrix"
        x_group.attrs["encoding-version"] = "0.1.0"
        x_group.attrs["shape"] = np.array([n_rows, n_vars], dtype=np.int64)

        chunk_elems = 1_000_000
        data_ds = x_group.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(chunk_elems,), compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )
        indices_ds = x_group.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(chunk_elems,), compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )
        x_group.create_dataset(
            "indptr", data=indptr, dtype=np.int64,
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )

        write_pos = 0
        t_start = time.time()
        for ctx in CONTEXTS:
            _, nz_idx, nz_val = context_means[ctx]
            for batch_start in range(0, n_genes, WRITE_BATCH_GENES):
                batch_genes = genes[batch_start:batch_start + WRITE_BATCH_GENES]
                n_batch_rows = len(batch_genes) * cells_per_pert

                batch_indices = np.tile(nz_idx, n_batch_rows)
                batch_data = np.tile(nz_val, n_batch_rows)
                n_new = batch_data.shape[0]

                data_ds.resize((write_pos + n_new,))
                indices_ds.resize((write_pos + n_new,))
                data_ds[write_pos:write_pos + n_new] = batch_data
                indices_ds[write_pos:write_pos + n_new] = batch_indices
                write_pos += n_new

            elapsed = time.time() - t_start
            print(f"  context {ctx}: wrote {n_genes * cells_per_pert} rows "
                  f"({row_nnz_per_context[ctx]} nnz/row) [{elapsed:.1f}s elapsed]")

        assert write_pos == total_nnz, f"wrote {write_pos} nnz, expected {total_nnz}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-pert", type=int, default=DEFAULT_CELLS_PER_PERT)
    parser.add_argument("--genes-limit", type=int, default=None,
                         help="Only use the first N genes from pert_counts.csv (for smoke-testing).")
    parser.add_argument("--materialize", action="store_true",
                         help="Physically write every predicted row instead of using an HDF5 "
                              "virtual dataset. Produces a much larger file and takes much longer.")
    parser.add_argument("--min-cpm", type=float, default=DEFAULT_MIN_CPM,
                         help="Zero out genes below this counts-per-million in the mean vector, "
                              "to satisfy the VCC scoring server's density cap. Use 0 to disable.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Computing per-context control means (streamed, backed='r')...")
    context_means = {}
    var_names_ref = None
    for ctx in CONTEXTS:
        var_names, nz_idx, nz_val = compute_context_mean(CONTEXT_FILES[ctx], min_cpm=args.min_cpm)
        if var_names_ref is None:
            var_names_ref = var_names
        elif not np.array_equal(var_names_ref, var_names):
            raise ValueError(f"var_names for context {ctx} do not match context {CONTEXTS[0]}")
        context_means[ctx] = (var_names, nz_idx, nz_val)
        print(f"  context {ctx}: {len(nz_idx)} / {len(var_names)} nonzero mean entries")

    genes = load_target_genes(args.genes_limit)
    print(f"Target genes: {len(genes)} (from {PERT_COUNTS_PATH})")

    obs = build_obs(genes, args.cells_per_pert)
    n_rows = len(obs)
    n_vars = len(var_names_ref)
    print(f"Predicted matrix shape: {n_rows} cells x {n_vars} genes "
          f"({len(CONTEXTS)} contexts x {len(genes)} genes x {args.cells_per_pert} cells/gene)")

    print(f"Writing obs/var skeleton to {OUT_PATH} ...")
    write_placeholder(OUT_PATH, n_rows, var_names_ref, obs)

    if args.materialize:
        print("Materializing predicted X (context mean broadcast to every target cell)...")
        stream_write_x_materialized(OUT_PATH, genes, context_means, args.cells_per_pert, n_vars)
    else:
        print("Writing predicted X as an HDF5 virtual dataset (context mean broadcast to every target cell)...")
        stream_write_x_virtual(OUT_PATH, genes, context_means, args.cells_per_pert, n_vars)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Done. Wrote {OUT_PATH} ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
