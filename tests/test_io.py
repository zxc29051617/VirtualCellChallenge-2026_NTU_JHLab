from pathlib import Path

import h5py
import numpy as np
import pytest

from vcc2026.io import (
    control_gene_profile,
    control_profile_to_sparse,
    group_mean_profiles,
    read_name_list,
    read_var_names,
    validate_gene_alignment,
)


def _write_dense_h5ad(path: Path, counts: np.ndarray, perturbations: list[str]) -> None:
    """Minimal h5ad in the layout scPerturb uses: dense X, plain var index."""
    categories = sorted(set(perturbations))
    codes = np.array([categories.index(p) for p in perturbations], dtype=np.int16)
    with h5py.File(path, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        x = handle.create_dataset("X", data=counts.astype(np.float32), chunks=(2, 2))
        x.attrs["encoding-type"] = "array"
        x.attrs["encoding-version"] = "0.2.0"

        obs = handle.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        obs.attrs["_index"] = "cell_barcode"
        obs.create_dataset(
            "cell_barcode",
            data=np.array([f"c{i}" for i in range(len(perturbations))], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        pert = obs.create_group("perturbation")
        pert.attrs["encoding-type"] = "categorical"
        pert.attrs["encoding-version"] = "0.2.0"
        pert.create_dataset(
            "categories", data=np.array(categories, dtype=object), dtype=h5py.string_dtype("utf-8")
        )
        pert.create_dataset("codes", data=codes)

        var = handle.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        var.attrs["_index"] = "gene_name"
        var.create_dataset(
            "gene_name",
            data=np.array([f"g{i}" for i in range(counts.shape[1])], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )


def test_group_mean_profiles_averages_per_category(tmp_path: Path) -> None:
    path = tmp_path / "dense.h5ad"
    counts = np.array([[1, 0, 3], [3, 0, 5], [10, 2, 0], [0, 0, 0], [4, 4, 4]])
    _write_dense_h5ad(path, counts, ["A", "A", "B", "B", "C"])

    # row_chunk_size below the chunk height forces multiple blocks and a partial last block
    means, categories, cell_counts = group_mean_profiles(path, "perturbation", row_chunk_size=2)

    assert categories == ["A", "B", "C"]
    assert cell_counts.tolist() == [2, 2, 1]
    assert means[0].tolist() == [2.0, 0.0, 4.0]
    assert means[1].tolist() == [5.0, 1.0, 0.0]
    assert means[2].tolist() == [4.0, 4.0, 4.0]


def test_read_var_names_handles_plain_index_dataset(tmp_path: Path) -> None:
    path = tmp_path / "dense.h5ad"
    _write_dense_h5ad(path, np.zeros((2, 3)), ["A", "B"])
    assert read_var_names(path) == ["g0", "g1", "g2"]


def test_read_name_list_strips_header(tmp_path: Path) -> None:
    path = tmp_path / "genes.csv"
    path.write_text("gene_name\nA\nB\n", encoding="utf-8")
    assert read_name_list(path) == ["A", "B"]


def test_validate_gene_alignment_reports_first_mismatch() -> None:
    mismatch = validate_gene_alignment(["A", "B", "C"], ["A", "X", "C"])
    assert mismatch and "first mismatch at 1" in mismatch[0]


def test_control_gene_profile_streams_sparse_matrix(tmp_path: Path) -> None:
    path = tmp_path / "toy.h5ad"
    with h5py.File(path, "w") as handle:
        handle.attrs["encoding-type"] = "anndata"
        handle.attrs["encoding-version"] = "0.1.0"
        x = handle.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.array([2, 3], dtype=np.int64)
        x.create_dataset("data", data=np.array([1, 2, 3], dtype=np.float32))
        x.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
        x.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
        for group_name in ["layers", "obsm", "obsp", "uns", "varm", "varp"]:
            handle.create_group(group_name)
        obs = handle.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        index = obs.create_group("_index")
        index.attrs["encoding-type"] = "nullable-string-array"
        index.attrs["encoding-version"] = "0.1.0"
        index.attrs["na-value"] = "NaN"
        index.create_dataset("mask", data=np.zeros(2, dtype=np.bool_))
        index.create_dataset("values", data=np.array(["c1", "c2"], dtype=object), dtype=h5py.string_dtype("utf-8"))
        ctx = obs.create_group("context")
        ctx.attrs["encoding-type"] = "categorical"
        ctx.attrs["encoding-version"] = "0.2.0"
        ctx.create_dataset("categories", data=np.array(["A"], dtype=object), dtype=h5py.string_dtype("utf-8"))
        ctx.create_dataset("codes", data=np.array([0, 0], dtype=np.int8))
        tg = obs.create_group("target_gene")
        tg.attrs["encoding-type"] = "categorical"
        tg.attrs["encoding-version"] = "0.2.0"
        tg.create_dataset("categories", data=np.array(["non-targeting"], dtype=object), dtype=h5py.string_dtype("utf-8"))
        tg.create_dataset("codes", data=np.array([0, 0], dtype=np.int8))
        var = handle.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        index = var.create_group("_index")
        index.attrs["encoding-type"] = "nullable-string-array"
        index.attrs["encoding-version"] = "0.1.0"
        index.attrs["na-value"] = "NaN"
        index.create_dataset("mask", data=np.zeros(3, dtype=np.bool_))
        index.create_dataset("values", data=np.array(["g1", "g2", "g3"], dtype=object), dtype=h5py.string_dtype("utf-8"))

    mean, mean_library, n_cells = control_gene_profile(path, row_chunk_size=1)
    assert n_cells == 2
    assert mean_library == 3.0
    assert mean.tolist() == [0.5, 1.5, 1.0]


def test_profile_to_sparse_filters_by_cpm_before_rounding() -> None:
    """Order matters: a gene under the CPM floor goes even though it would have rounded up."""
    # 0.6 counts in a million-count profile is 0.6 CPM, but rounds to 1.
    counts = np.array([1_000_000.0, 0.6])
    assert control_profile_to_sparse(counts, min_cpm=1.0).gene_indices.tolist() == [0]
    assert control_profile_to_sparse(counts, min_cpm=0.0).gene_indices.tolist() == [0, 1]


def test_profile_to_sparse_keeps_means_unrounded() -> None:
    """Selection uses the half-count threshold; the values stay as means for the writer."""
    counts = np.array([10.4, 10.6, 0.4, 0.6])
    kept = control_profile_to_sparse(counts, min_cpm=0.0)
    assert kept.gene_indices.tolist() == [0, 1, 3]
    assert kept.gene_values == pytest.approx([10.4, 10.6, 0.6])


def test_max_genes_caps_to_the_strongest() -> None:
    counts = np.array([5.0, 50.0, 500.0, 2.0])
    kept = control_profile_to_sparse(counts, min_cpm=0.0, max_genes=2)
    assert sorted(kept.gene_indices.tolist()) == [1, 2]
