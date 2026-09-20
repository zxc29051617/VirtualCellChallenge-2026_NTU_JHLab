from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


def _categorical(group: h5py.Group, name: str, categories: list[str], codes: np.ndarray) -> None:
    node = group.create_group(name)
    node.attrs["encoding-type"] = "categorical"
    node.attrs["encoding-version"] = "0.2.0"
    node.attrs["ordered"] = np.bool_(False)
    node.create_dataset(
        "categories", data=np.asarray(categories, dtype=object), dtype=h5py.string_dtype("utf-8")
    )
    node.create_dataset("codes", data=codes, compression="gzip", compression_opts=4)


def cells_from_mean(
    values: np.ndarray, n_cells: int, rng: np.random.Generator
) -> np.ndarray:
    """`n_cells` integer rows whose column means are `values`.

    A cell must hold whole counts, but the metrics read group-summed profiles, so a mean of
    5.3 is not a 5 -- it is 70% of cells at 5 and 30% at 6. Rounding each gene to an integer
    instead would silently discard every predicted change smaller than half a count, which
    on this data is most of them.

    Each gene's surplus cells are drawn independently, so the cells differ from one another
    in the way real ones do rather than splitting into two rigid groups.
    """

    floor = np.floor(values)
    surplus = np.rint((values - floor) * n_cells).astype(np.int64)
    cells = np.tile(floor.astype(np.float32), (n_cells, 1))
    for gene, count in enumerate(surplus):
        if count >= n_cells:
            cells[:, gene] += 1
        elif count > 0:
            cells[rng.choice(n_cells, size=int(count), replace=False), gene] += 1
    return cells


@dataclass(frozen=True)
class ContextPrediction:
    """Target mean counts per perturbation for one context, over a shared set of genes.

    `means` is [n_perturbations, len(gene_indices)]. A single row broadcasts to every
    perturbation, which is what a context-mean baseline wants; more rows let each
    perturbation carry its own prediction.
    """

    context: str
    gene_indices: np.ndarray
    means: np.ndarray

    @classmethod
    def from_baseline(cls, baseline: ContextBaseline) -> ContextPrediction:
        profile = baseline.profile.tocsr()
        return cls(
            context=baseline.context,
            gene_indices=np.asarray(profile.indices, dtype=np.int32),
            means=np.asarray(profile.data, dtype=np.float32).reshape(1, -1),
        )

    def row(self, perturbation: int) -> np.ndarray:
        return self.means[perturbation if self.means.shape[0] > 1 else 0]


def _expand(
    profile: sparse.csr_matrix, n_cells: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten one context's cells into (indices, data, indptr) over those cells."""
    columns = np.asarray(profile.indices, dtype=np.int32)
    cells = cells_from_mean(np.asarray(profile.data, dtype=np.float64), n_cells, rng)

    present = cells > 0
    per_row = present.sum(axis=1).astype(np.int64)
    indices = np.concatenate([columns[row] for row in present])
    data = np.concatenate([cells[i][row] for i, row in enumerate(present)])
    return (
        indices.astype(np.int32, copy=False),
        data.astype(np.float32, copy=False),
        np.concatenate(([0], np.cumsum(per_row))).astype(np.int64),
    )


def _write_x_virtual(
    handle: h5py.File,
    x_group: h5py.Group,
    contexts: list[str],
    expanded: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    perturbations: int,
    indptr: np.ndarray,
) -> None:
    """Store one context's cells once and map every perturbation onto the same cells.

    The prediction does not depend on which gene is knocked down, so all `perturbations`
    copies of a context are the same `cells_per_pert` rows. Storing them once and mapping
    keeps the file from growing with the perturbation count; gzip cannot do the same,
    because its 32 KB window is smaller than the block it would have to see repeat.

    Technique taken from scnpg/VCC_2026_NTU_JHLab.
    """

    store = handle["uns"].create_group("_context_cells")
    sources = {}
    for ctx in contexts:
        indices, data, _ = expanded[ctx]
        index_node = store.create_dataset(f"{ctx}_indices", data=indices)
        value_node = store.create_dataset(f"{ctx}_data", data=data)
        for node in (index_node, value_node):
            node.attrs["encoding-type"] = "array"
            node.attrs["encoding-version"] = "0.2.0"
        sources[ctx] = (h5py.VirtualSource(index_node), h5py.VirtualSource(value_node))

    total_nnz = int(indptr[-1])
    data_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.float32)
    indices_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.int32)

    position = 0
    for ctx in contexts:
        index_source, value_source = sources[ctx]
        cell_indptr = expanded[ctx][2]
        n_cells = cell_indptr.size - 1
        for _ in range(perturbations):
            for cell in range(n_cells):
                start, stop = int(cell_indptr[cell]), int(cell_indptr[cell + 1])
                span = stop - start
                data_layout[position: position + span] = value_source[start:stop]
                indices_layout[position: position + span] = index_source[start:stop]
                position += span

    x_group.create_virtual_dataset("data", data_layout)
    x_group.create_virtual_dataset("indices", indices_layout)
    x_group.create_dataset("indptr", data=indptr, dtype=np.int64)


def _write_x_materialized(
    x_group: h5py.Group,
    contexts: list[str],
    expanded: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    perturbations: int,
    indptr: np.ndarray,
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
        ctx_indices, ctx_data, _ = expanded[ctx]
        span = ctx_data.size
        for _ in range(perturbations):
            data[cursor: cursor + span] = ctx_data
            indices[cursor: cursor + span] = ctx_indices
            cursor += span


def _write_x_per_perturbation(
    handle: h5py.File,
    x_group: h5py.Group,
    contexts: list[str],
    predictions: Mapping[str, ContextPrediction],
    n_perturbations: int,
    cells_per_pert: int,
    cell_patterns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Write a prediction that differs per perturbation.

    Every row is now distinct, so the virtual trick cannot collapse them the way it does for
    a context-mean baseline. Two things still keep this bounded. The gene set is shared, so
    `indices` is stored once and mapped everywhere. And a perturbation's cells are drawn from
    `cell_patterns` distinct rows rather than `cells_per_pert`, which caps stored values at
    perturbations x patterns instead of perturbations x cells; the cost is that a mean is
    represented to 1/patterns of a count rather than 1/cells.
    """

    n_genes = predictions[contexts[0]].gene_indices.size
    per_row = np.full(len(contexts) * n_perturbations * cells_per_pert, n_genes, dtype=np.int64)
    indptr = np.concatenate(([0], np.cumsum(per_row))).astype(np.int64)

    store = handle["uns"].create_group("_context_cells")
    shared_indices = store.create_dataset(
        "gene_indices", data=predictions[contexts[0]].gene_indices.astype(np.int32)
    )
    shared_indices.attrs["encoding-type"] = "array"
    shared_indices.attrs["encoding-version"] = "0.2.0"

    stored = {}
    for ctx in contexts:
        prediction = predictions[ctx]
        rows = np.empty((n_perturbations * cell_patterns, n_genes), dtype=np.float32)
        for p in range(n_perturbations):
            rows[p * cell_patterns: (p + 1) * cell_patterns] = cells_from_mean(
                prediction.row(p).astype(np.float64), cell_patterns, rng
            )
        node = store.create_dataset(
            f"{ctx}_data", data=rows.reshape(-1), compression="gzip", compression_opts=4
        )
        node.attrs["encoding-type"] = "array"
        node.attrs["encoding-version"] = "0.2.0"
        stored[ctx] = h5py.VirtualSource(node)

    total_nnz = int(indptr[-1])
    data_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.float32)
    indices_layout = h5py.VirtualLayout(shape=(total_nnz,), dtype=np.int32)
    index_source = h5py.VirtualSource(shared_indices)

    position = 0
    for ctx in contexts:
        source = stored[ctx]
        for p in range(n_perturbations):
            for cell in range(cells_per_pert):
                offset = (p * cell_patterns + cell % cell_patterns) * n_genes
                data_layout[position: position + n_genes] = source[offset: offset + n_genes]
                indices_layout[position: position + n_genes] = index_source
                position += n_genes

    x_group.create_virtual_dataset("data", data_layout)
    x_group.create_virtual_dataset("indices", indices_layout)
    x_group.create_dataset("indptr", data=indptr, dtype=np.int64)
    return indptr


def write_prediction_h5ad(
    output_path: str | Path,
    *,
    gene_names: list[str],
    perturbations: list[str],
    baselines: Mapping[str, ContextBaseline],
    cells_per_pert: int = 400,
    virtual: bool = True,
    seed: int = 0,
) -> Path:
    contexts = list(baselines.keys())
    rows_per_context = len(perturbations) * cells_per_pert
    n_rows = len(contexts) * rows_per_context
    n_genes = len(gene_names)

    rng = np.random.default_rng(seed)
    expanded = {
        ctx: _expand(baselines[ctx].profile.tocsr(), cells_per_pert, rng) for ctx in contexts
    }
    indptr = np.concatenate(
        ([0], np.cumsum(np.concatenate([
            np.tile(np.diff(expanded[ctx][2]), len(perturbations)) for ctx in contexts
        ])))
    ).astype(np.int64)

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
            _write_x_virtual(handle, x_group, contexts, expanded, len(perturbations), indptr)
        else:
            _write_x_materialized(
                x_group, contexts, expanded, len(perturbations), indptr
            )

        _write_frame(handle, gene_names, perturbations, contexts, cells_per_pert)

    return output


def _write_frame(
    handle: h5py.File,
    gene_names: list[str],
    perturbations: list[str],
    contexts: list[str],
    cells_per_pert: int,
) -> None:
    """obs and var, in the order X's rows are written: context, then perturbation, then cell."""

    rows_per_context = len(perturbations) * cells_per_pert
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


def write_perturbation_h5ad(
    output_path: str | Path,
    *,
    gene_names: list[str],
    perturbations: list[str],
    predictions: Mapping[str, ContextPrediction],
    cells_per_pert: int = 400,
    cell_patterns: int = 100,
    seed: int = 0,
) -> Path:
    """Write a prediction that differs per perturbation, rather than one profile per context."""

    contexts = list(predictions.keys())
    n_rows = len(contexts) * len(perturbations) * cells_per_pert
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
        x_group.attrs["shape"] = np.array([n_rows, len(gene_names)], dtype=np.int64)

        _write_x_per_perturbation(
            handle,
            x_group,
            contexts,
            predictions,
            len(perturbations),
            cells_per_pert,
            cell_patterns,
            np.random.default_rng(seed),
        )
        _write_frame(handle, gene_names, perturbations, contexts, cells_per_pert)

    return output
