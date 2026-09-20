from pathlib import Path

import h5py
import pytest
from scipy import sparse

from vcc2026.baseline import ContextBaseline
from vcc2026.submission import write_prediction_h5ad


def _baseline() -> ContextBaseline:
    return ContextBaseline(
        context="A",
        profile=sparse.csr_matrix(([1.0, 2.0], ([0, 0], [0, 2])), shape=(1, 3)),
        n_control_cells=2,
        mean_library_size=3.0,
    )


@pytest.mark.parametrize("virtual", [True, False])
def test_write_prediction_h5ad_creates_readable_h5ad(tmp_path: Path, virtual: bool) -> None:
    output = tmp_path / "pred.h5ad"
    write_prediction_h5ad(
        output,
        gene_names=["g1", "g2", "g3"],
        perturbations=["p1", "p2"],
        baselines={"A": _baseline()},
        cells_per_pert=2,
        block_rows=1,
        virtual=virtual,
    )
    with h5py.File(output, "r") as handle:
        assert tuple(int(x) for x in handle["X"].attrs["shape"]) == (4, 3)
        assert handle["obs/context/codes"].shape == (4,)
        assert handle["obs/target_gene/codes"].shape == (4,)
        assert handle["X/data"].shape[0] == 8
        assert handle["X/indptr"].shape[0] == 5


def test_virtual_and_materialized_hold_identical_values(tmp_path: Path) -> None:
    """The virtual file must read back exactly what a physically written one does."""
    written = {}
    for virtual in (True, False):
        path = tmp_path / f"pred_{virtual}.h5ad"
        write_prediction_h5ad(
            path,
            gene_names=["g1", "g2", "g3"],
            perturbations=["p1", "p2"],
            baselines={"A": _baseline()},
            cells_per_pert=2,
            virtual=virtual,
        )
        with h5py.File(path, "r") as handle:
            written[virtual] = (
                handle["X/data"][()],
                handle["X/indices"][()],
                handle["X/indptr"][()],
            )

    for actual, expected in zip(written[True], written[False]):
        assert (actual == expected).all()


def test_virtual_stores_one_copy_per_context_regardless_of_row_count(tmp_path: Path) -> None:
    """The invariant behind the technique: stored bytes do not grow with the row count.

    Whether that makes the file smaller than gzip depends on row width and how compressible
    the counts are -- at 1,000 genes per row of real rounded means, gzip is smaller. The
    property that always holds, and that matters as the gene count rises, is this one.
    """

    stored = {}
    for cells_per_pert in (2, 200):
        path = tmp_path / f"rows_{cells_per_pert}.h5ad"
        write_prediction_h5ad(
            path,
            gene_names=["g1", "g2", "g3"],
            perturbations=["p1", "p2"],
            baselines={"A": _baseline()},
            cells_per_pert=cells_per_pert,
        )
        with h5py.File(path, "r") as handle:
            source = handle["uns/_context_profile"]
            stored[cells_per_pert] = source["A_data"].shape[0]
            assert handle["X/data"].shape[0] == 2 * 2 * cells_per_pert

    assert stored[2] == stored[200] == 2

