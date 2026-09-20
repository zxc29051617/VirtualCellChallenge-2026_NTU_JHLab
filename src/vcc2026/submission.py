from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

from .baseline import ContextBaseline


def _write_string_dataset(group: h5py.Group, name: str, values: list[str], *, chunks: int = 8192) -> None:
    dtype = h5py.string_dtype("utf-8")
    dataset = group.create_dataset(
        name,
        shape=(len(values),),
        dtype=dtype,
        chunks=(min(chunks, max(1, len(values))),),
        compression="gzip",
        compression_opts=4,
    )
    if values:
        dataset[:] = np.asarray(values, dtype=object)


def _create_nullable_string_array(group: h5py.Group, values: list[str], name: str) -> None:
    container = group.create_group(name)
    container.attrs["encoding-type"] = "nullable-string-array"
    container.attrs["encoding-version"] = "0.1.0"
    container.attrs["na-value"] = "NaN"
    mask = container.create_dataset(
        "mask",
        data=np.zeros(len(values), dtype=np.bool_),
        chunks=(min(8192, max(1, len(values))),),
        compression="gzip",
        compression_opts=4,
    )
    mask.attrs["encoding-type"] = "array"
    mask.attrs["encoding-version"] = "0.2.0"
    _write_string_dataset(container, "values", values)


def _categorical(group: h5py.Group, name: str, categories: list[str], codes: np.ndarray) -> None:
    node = group.create_group(name)
    node.attrs["encoding-type"] = "categorical"
    node.attrs["encoding-version"] = "0.2.0"
    node.attrs["ordered"] = np.bool_(False)
    node.create_dataset(
        "categories", data=np.asarray(categories, dtype=object), dtype=h5py.string_dtype("utf-8")
    )
    node.create_dataset("codes", data=codes, compression="gzip", compression_opts=4)


def _indptr(
    contexts: list[str],
    baselines: Mapping[str, ContextBaseline],
    rows_per_context: int,
) -> np.ndarray:
    per_row = np.concatenate(
        [
            np.full(rows_per_context, int(baselines[ctx].profile.nnz), dtype=np.int64)
            for ctx in contexts
        ]
    )
    return np.concatenate(([0], np.cumsum(per_row))).astype(np.int64)


def _write_x_virtual(
    handle: h5py.File,
    x_group: h5py.Group,
    contexts: list[str],
    baselines: Mapping[str, ContextBaseline],
    rows_per_context: int,
    indptr: np.ndarray,
) -> None:
    """Store each context's row once and map every predicted row onto that single copy.

    Every row of a context-mean prediction is byte-for-byte identical, but gzip cannot
    exploit that: its 32 KB DEFLATE window is smaller than one row, so it never sees the
    repetition. An HDF5 virtual dataset expands on read instead, so `.X` returns exactly
    what a materialized file would while the file itself stays small.

    Technique taken from scnpg/VCC_2026_NTU_JHLab.
    """

    total_nnz = int(indptr[-1])
    store = handle["uns"].create_group("_context_profile")
    sources = {}
    for ctx in contexts:
        profile = baselines[ctx].profile.tocsr()
        indices = store.create_dataset(
            f"{ctx}_indices", data=np.asarray(profile.indices, dtype=np.int32)
        )
        values = store.create_dataset(
            f"{ctx}_data", data=np.asarray(profile.data, dtype=np.float32)
        )
        for node in (indices, values):
            node.attrs["encoding-type"] = "array"
            node.attrs["encoding-version"] = "0.2.0"
        sources[ctx] = (h5py.VirtualSource(indices), h5py.VirtualSource(values))

    data_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.float32)
    indices_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.int32)

    position = 0
    for ctx in contexts:
        index_source, value_source = sources[ctx]
        nnz = int(baselines[ctx].profile.nnz)
        for _ in range(rows_per_context):
            data_layout[position: position + nnz] = value_source
            indices_layout[position: position + nnz] = index_source
            position += nnz

    x_group.create_virtual_dataset("data", data_layout)
    x_group.create_virtual_dataset("indices", indices_layout)
    x_group.create_dataset("indptr", data=indptr, dtype=np.int64)


def _write_x_materialized(
    x_group: h5py.Group,
    contexts: list[str],
    baselines: Mapping[str, ContextBaseline],
    rows_per_context: int,
    indptr: np.ndarray,
    block_rows: int,
) -> None:
    """Physically write every row. Larger and slower; for tools that reject virtual datasets."""

    total_nnz = int(indptr[-1])
    chunk = (min(1_000_000, max(1, total_nnz)),)
    data = x_group.create_dataset(
        "data", shape=(total_nnz,), dtype=np.float32, chunks=chunk,
        compression="gzip", compression_opts=4,
    )
    indices = x_group.create_dataset(
        "indices", shape=(total_nnz,), dtype=np.int32, chunks=chunk,
        compression="gzip", compression_opts=4,
    )
    x_group.create_dataset(
        "indptr", data=indptr, dtype=np.int64,
        chunks=(min(8192, max(1, indptr.size)),), compression="gzip", compression_opts=4,
    )

    cursor = 0
    for ctx in contexts:
        profile = baselines[ctx].profile.tocsr()
        row_data = np.asarray(profile.data, dtype=np.float32)
        row_indices = np.asarray(profile.indices, dtype=np.int32)
        nnz = int(row_data.size)
        for start in range(0, rows_per_context, block_rows):
            rows = min(block_rows, rows_per_context - start)
            span = nnz * rows
            data[cursor: cursor + span] = np.tile(row_data, rows)
            indices[cursor: cursor + span] = np.tile(row_indices, rows)
            cursor += span


def write_prediction_h5ad(
    output_path: str | Path,
    *,
    gene_names: list[str],
    perturbations: list[str],
    baselines: Mapping[str, ContextBaseline],
    cells_per_pert: int = 400,
    block_rows: int = 256,
    virtual: bool = True,
) -> Path:
    contexts = list(baselines.keys())
    rows_per_context = len(perturbations) * cells_per_pert
    n_rows = len(contexts) * rows_per_context
    n_genes = len(gene_names)
    indptr = _indptr(contexts, baselines, rows_per_context)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"

        for name in ["layers", "obsm", "obsp", "uns", "varm", "varp"]:
            empty = handle.create_group(name)
            empty.attrs["encoding-type"] = "dict"
            empty.attrs["encoding-version"] = "0.1.0"

        x_group = handle.create_group("X")
        x_group.attrs["encoding-type"] = "csr_matrix"
        x_group.attrs["encoding-version"] = "0.1.0"
        x_group.attrs["shape"] = np.array([n_rows, n_genes], dtype=np.int64)

        if virtual:
            _write_x_virtual(handle, x_group, contexts, baselines, rows_per_context, indptr)
        else:
            _write_x_materialized(
                x_group, contexts, baselines, rows_per_context, indptr, block_rows
            )

        obs_group = handle.create_group("obs")
        obs_group.attrs["encoding-type"] = "dataframe"
        obs_group.attrs["encoding-version"] = "0.2.0"
        obs_group.attrs["_index"] = "_index"
        obs_group.attrs["column-order"] = np.asarray(["target_gene", "context"], dtype=object)
        _create_nullable_string_array(
            obs_group,
            [
                f"{ctx}_{pert}_{idx:04d}"
                for ctx in contexts
                for pert in perturbations
                for idx in range(cells_per_pert)
            ],
            "_index",
        )
        _categorical(
            obs_group,
            "context",
            contexts,
            np.repeat(np.arange(len(contexts), dtype=np.int8), rows_per_context),
        )
        _categorical(
            obs_group,
            "target_gene",
            list(perturbations),
            np.tile(
                np.repeat(np.arange(len(perturbations), dtype=np.int16), cells_per_pert),
                len(contexts),
            ),
        )

        var_group = handle.create_group("var")
        var_group.attrs["encoding-type"] = "dataframe"
        var_group.attrs["encoding-version"] = "0.2.0"
        var_group.attrs["_index"] = "_index"
        var_group.attrs["column-order"] = np.asarray([], dtype=np.float64)
        _create_nullable_string_array(var_group, gene_names, "_index")

    return output
