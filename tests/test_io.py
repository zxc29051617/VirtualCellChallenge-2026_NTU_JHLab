from pathlib import Path

import h5py
import numpy as np

from vcc2026.io import control_gene_profile, read_name_list, validate_gene_alignment


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
    assert mean.tolist() == [0.0, 2.0, 1.0]
