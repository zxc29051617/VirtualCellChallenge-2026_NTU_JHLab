from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SparseProfile:
    gene_indices: np.ndarray
    gene_values: np.ndarray

    @property
    def nnz(self) -> int:
        return int(self.gene_indices.size)

    def as_csr_row(self, n_genes: int) -> sparse.csr_matrix:
        indptr = np.array([0, self.nnz], dtype=np.int64)
        return sparse.csr_matrix(
            (self.gene_values.astype(np.float32, copy=False), self.gene_indices.astype(np.int32, copy=False), indptr),
            shape=(1, n_genes),
        )


def _decode_many(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def read_name_list(path: str | Path) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    first = lines[0].lower()
    if first in {"gene_name", "gene", "target_gene", "perturbation", "target"}:
        return lines[1:]
    return lines


def read_csv_column(path: str | Path, expected_header: str | None = None) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    if expected_header is not None and lines[0].lower() == expected_header.lower():
        return lines[1:]
    if len(lines) > 1 and "," in lines[0]:
        header = lines[0].split(",")[0].strip().lower()
        if expected_header is not None and header == expected_header.lower():
            return [line.split(",")[0].strip() for line in lines[1:]]
    return [line.split(",")[0].strip() for line in lines]


def control_gene_profile(
    path: str | Path,
    *,
    row_chunk_size: int = 1024,
) -> tuple[np.ndarray, float, int]:
    """Return mean counts per gene for the control file.

    The mean is returned unrounded. Rounding to the integer counts a submission must carry
    is `control_profile_to_sparse`'s job, because it has to happen after the low-expression
    filter rather than before it.

    The function streams the sparse matrix from disk to avoid loading the full
    control matrix into memory.
    """

    with h5py.File(path, "r") as handle:
        shape = tuple(int(x) for x in handle["X"].attrs["shape"])
        n_rows, n_genes = shape
        indptr = handle["X/indptr"][()]
        gene_sums = np.zeros(n_genes, dtype=np.float64)
        total_counts = 0.0
        for start in range(0, n_rows, row_chunk_size):
            stop = min(n_rows, start + row_chunk_size)
            lo = int(indptr[start])
            hi = int(indptr[stop])
            chunk_indices = handle["X/indices"][lo:hi]
            chunk_data = handle["X/data"][lo:hi]
            gene_sums += np.bincount(chunk_indices, weights=chunk_data, minlength=n_genes)
            total_counts += float(chunk_data.sum())
        mean_counts = gene_sums / float(n_rows)
        return mean_counts.astype(np.float32), float(total_counts / float(n_rows)), n_rows


def control_profile_to_sparse(
    mean_counts: np.ndarray,
    *,
    min_cpm: float = 1.0,
    max_genes: int | None = None,
) -> SparseProfile:
    """Round a mean profile to the integer counts a submission must carry.

    Genes below `min_cpm` counts per million are dropped first. They carry almost no
    expression mass, but each would cost a stored entry against the scoring density cap.
    Rounding then removes whatever averages below half a count. Filtering before rounding
    rather than after is what keeps the surviving set defined by expression rather than by
    an arbitrary rank cut.

    `max_genes` caps the result to the strongest genes. It costs real signal -- on the
    challenge controls, a cap of 1,000 discards 6,608 genes carrying 57% of the expression
    -- so leave it unset unless something downstream demands a hard bound.
    """

    counts = np.asarray(mean_counts, dtype=np.float64)
    total = counts.sum()
    if min_cpm > 0 and total > 0:
        counts = np.where(counts * (1e6 / total) >= min_cpm, counts, 0.0)

    rounded = np.rint(counts).astype(np.float32)
    gene_indices = np.flatnonzero(rounded)
    gene_values = rounded[gene_indices]
    if max_genes is not None and gene_indices.size > max_genes:
        order = np.argsort(-gene_values, kind="stable")[:max_genes]
        gene_indices = gene_indices[order]
        gene_values = gene_values[order]
    return SparseProfile(gene_indices=gene_indices.astype(np.int32, copy=False), gene_values=gene_values)


def validate_gene_alignment(expected: list[str], observed: list[str]) -> list[str]:
    if expected == observed:
        return []
    mismatches: list[str] = []
    if len(expected) != len(observed):
        mismatches.append(f"length mismatch: expected={len(expected)} observed={len(observed)}")
    for idx, (exp, obs) in enumerate(zip(expected, observed)):
        if exp != obs:
            mismatches.append(f"first mismatch at {idx}: expected={exp} observed={obs}")
            break
    return mismatches


def load_control_gene_names(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return _decode_many(handle["var/_index/values"][()])


def load_control_context(path: str | Path) -> str:
    with h5py.File(path, "r") as handle:
        values = _decode_many(handle["obs/context/categories"][()])
        if len(values) != 1:
            raise ValueError(f"expected exactly one context in control file, found {values}")
        return values[0]


def load_control_stats(path: str | Path) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        n_rows, n_genes = (int(x) for x in handle["X"].attrs["shape"])
        return {"n_cells": n_rows, "n_genes": n_genes}


def _index_names(frame: h5py.Group) -> list[str]:
    name = frame.attrs.get("_index", "_index")
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    node = frame[name]
    values = node["values"][()] if isinstance(node, h5py.Group) else node[()]
    return _decode_many(values)


def read_var_names(path: str | Path) -> list[str]:
    """Gene names from any h5ad, whether var's index is a dataset or a nullable-string group."""
    with h5py.File(path, "r") as handle:
        return _index_names(handle["var"])


def read_categorical(path: str | Path, column: str) -> tuple[np.ndarray, list[str]]:
    """Return (codes, categories) for a categorical obs column."""
    with h5py.File(path, "r") as handle:
        group = handle[f"obs/{column}"]
        if not isinstance(group, h5py.Group) or "categories" not in group:
            raise ValueError(f"obs/{column} in {path} is not categorical")
        return group["codes"][()], _decode_many(group["categories"][()])


def group_mean_profiles(
    path: str | Path,
    column: str,
    *,
    row_chunk_size: int = 2048,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Mean expression per category of an obs column, streaming a dense X from disk.

    Returns (means[n_categories, n_genes], categories, cell_counts).

    The Replogle/scPerturb files store X densely, so reading one whole would allocate
    roughly 9-11 GB. This reads row blocks instead.
    """

    codes, categories = read_categorical(path, column)
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        if not isinstance(matrix, h5py.Dataset):
            raise TypeError(f"X in {path} is not a dense dataset")
        n_rows, n_genes = matrix.shape

        # Align reads to the HDF5 chunk grid, or each chunk is decompressed twice.
        chunk_rows = matrix.chunks[0] if matrix.chunks else row_chunk_size
        step = max(chunk_rows, (row_chunk_size // chunk_rows) * chunk_rows)

        sums = np.zeros((len(categories), n_genes), dtype=np.float64)
        counts = np.zeros(len(categories), dtype=np.int64)
        for start in range(0, n_rows, step):
            block = matrix[start: min(n_rows, start + step)]
            block_codes = codes[start: start + block.shape[0]]
            order = np.argsort(block_codes, kind="stable")
            ordered = block_codes[order]
            run_starts = np.concatenate(([0], np.flatnonzero(np.diff(ordered)) + 1))
            sums[ordered[run_starts]] += np.add.reduceat(block[order], run_starts, axis=0)
            counts += np.bincount(block_codes, minlength=len(categories))

    means = np.zeros_like(sums, dtype=np.float32)
    seen = counts > 0
    means[seen] = (sums[seen] / counts[seen, None]).astype(np.float32)
    return means, categories, counts
