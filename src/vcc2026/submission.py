from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

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


def write_prediction_h5ad(
    output_path: str | Path,
    *,
    gene_names: list[str],
    perturbations: list[str],
    baselines: Mapping[str, ContextBaseline],
    cells_per_pert: int = 400,
    block_rows: int = 256,
) -> Path:
    contexts = list(baselines.keys())
    n_rows = len(contexts) * len(perturbations) * cells_per_pert
    n_genes = len(gene_names)
    total_nnz = sum(int(baselines[ctx].profile.nnz) * len(perturbations) * cells_per_pert for ctx in contexts)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"

        x_group = handle.create_group("X")
        x_group.attrs["encoding-type"] = "csr_matrix"
        x_group.attrs["encoding-version"] = "0.1.0"
        x_group.attrs["shape"] = np.array([n_rows, n_genes], dtype=np.int64)

        data_ds = x_group.create_dataset(
            "data",
            shape=(total_nnz,),
            dtype=np.float32,
            chunks=(min(1_000_000, max(1, total_nnz)),),
            compression="gzip",
            compression_opts=4,
        )
        indices_ds = x_group.create_dataset(
            "indices",
            shape=(total_nnz,),
            dtype=np.int32,
            chunks=(min(1_000_000, max(1, total_nnz)),),
            compression="gzip",
            compression_opts=4,
        )
        indptr_ds = x_group.create_dataset(
            "indptr",
            shape=(n_rows + 1,),
            dtype=np.int64,
            chunks=(min(8192, max(1, n_rows + 1)),),
            compression="gzip",
            compression_opts=4,
        )

        row_cursor = 0
        nnz_cursor = 0
        indptr_ds[0] = 0

        obs_group = handle.create_group("obs")
        obs_group.attrs["encoding-type"] = "dataframe"
        obs_group.attrs["encoding-version"] = "0.2.0"
        obs_group.attrs["_index"] = "_index"
        obs_group.attrs["column-order"] = np.asarray(["target_gene", "context"], dtype=object)
        obs_index_values = [
            f"{ctx}_{pert}_{idx:04d}"
            for ctx in contexts
            for pert in perturbations
            for idx in range(cells_per_pert)
        ]
        _create_nullable_string_array(obs_group, obs_index_values, "_index")

        context_codes = np.empty(n_rows, dtype=np.int8)
        perturbation_codes = np.empty(n_rows, dtype=np.int16)
        context_categories = contexts
        pert_categories = list(perturbations)

        row_start = 0
        for ctx_code, context in enumerate(contexts):
            profile = baselines[context].profile.tocsr()
            row_data = np.asarray(profile.data, dtype=np.float32)
            row_indices = np.asarray(profile.indices, dtype=np.int32)
            row_nnz = int(row_data.size)
            for pert_code, _pert in enumerate(perturbations):
                for block_start in range(0, cells_per_pert, block_rows):
                    block = min(block_rows, cells_per_pert - block_start)
                    block_data = np.tile(row_data, block)
                    block_indices = np.tile(row_indices, block)
                    block_nnz = row_nnz * block
                    data_ds[nnz_cursor: nnz_cursor + block_nnz] = block_data
                    indices_ds[nnz_cursor: nnz_cursor + block_nnz] = block_indices
                    for offset in range(block):
                        row_cursor += 1
                        indptr_ds[row_cursor] = nnz_cursor + row_nnz * (offset + 1)
                    context_codes[row_start: row_start + block] = ctx_code
                    perturbation_codes[row_start: row_start + block] = pert_code
                    row_start += block
                    nnz_cursor += block_nnz
        context_group = obs_group.create_group("context")
        context_group.attrs["encoding-type"] = "categorical"
        context_group.attrs["encoding-version"] = "0.2.0"
        context_group.attrs["ordered"] = np.bool_(False)
        context_group.create_dataset("categories", data=np.asarray(context_categories, dtype=object), dtype=h5py.string_dtype("utf-8"))
        context_group.create_dataset("codes", data=context_codes, dtype=np.int8, compression="gzip", compression_opts=4)

        pert_group = obs_group.create_group("target_gene")
        pert_group.attrs["encoding-type"] = "categorical"
        pert_group.attrs["encoding-version"] = "0.2.0"
        pert_group.attrs["ordered"] = np.bool_(False)
        pert_group.create_dataset("categories", data=np.asarray(pert_categories, dtype=object), dtype=h5py.string_dtype("utf-8"))
        pert_group.create_dataset("codes", data=perturbation_codes, dtype=np.int16, compression="gzip", compression_opts=4)

        var_group = handle.create_group("var")
        var_group.attrs["encoding-type"] = "dataframe"
        var_group.attrs["encoding-version"] = "0.2.0"
        var_group.attrs["_index"] = "_index"
        var_group.attrs["column-order"] = np.asarray([], dtype=np.float64)
        _create_nullable_string_array(var_group, gene_names, "_index")

        for empty_name in ["layers", "obsm", "obsp", "uns", "varm", "varp"]:
            empty = handle.create_group(empty_name)
            empty.attrs["encoding-type"] = "dict"
            empty.attrs["encoding-version"] = "0.1.0"

    return output
