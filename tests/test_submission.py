from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

from vcc2026.baseline import ContextBaseline
from vcc2026.submission import (
    ContextPrediction,
    cells_from_mean,
    write_perturbation_h5ad,
    write_prediction_h5ad,
)


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
        virtual=virtual,
    )
    with h5py.File(output, "r") as handle:
        assert tuple(int(x) for x in handle["X"].attrs["shape"]) == (4, 3)
        assert handle["obs/context/codes"].shape == (4,)
        assert handle["obs/target_gene/codes"].shape == (4,)
        assert handle["X/data"].shape[0] == 8
        assert handle["X/indptr"].shape[0] == 5


def test_cells_from_mean_reproduces_a_fractional_mean_exactly(tmp_path: Path) -> None:
    """The point of varying cells: a mean of 5.3 is 70% of cells at 5 and 30% at 6."""
    rng = np.random.default_rng(0)
    cells = cells_from_mean(np.array([5.3, 0.6, 12.0]), 100, rng)

    assert cells.shape == (100, 3)
    assert (cells == np.floor(cells)).all(), "every cell must hold whole counts"
    assert cells[:, 0].mean() == pytest.approx(5.3)
    assert cells[:, 1].mean() == pytest.approx(0.6)
    assert cells[:, 2].mean() == pytest.approx(12.0)
    assert set(np.unique(cells[:, 0])) == {5.0, 6.0}


def test_written_cells_average_to_the_requested_profile(tmp_path: Path) -> None:
    """A fractional mean must survive into the file, not get rounded away on the way in."""
    output = tmp_path / "pred.h5ad"
    baseline = ContextBaseline(
        context="A",
        profile=sparse.csr_matrix(np.array([[5.3, 0.6, 12.0]], dtype=np.float32)),
        n_control_cells=2,
        mean_library_size=17.9,
    )
    write_prediction_h5ad(
        output,
        gene_names=["g1", "g2", "g3"],
        perturbations=["p1"],
        baselines={"A": baseline},
        cells_per_pert=100,
    )
    with h5py.File(output, "r") as handle:
        indptr = handle["X/indptr"][()]
        indices = handle["X/indices"][()]
        data = handle["X/data"][()]

    dense = np.zeros((100, 3))
    for row in range(100):
        lo, hi = int(indptr[row]), int(indptr[row + 1])
        dense[row, indices[lo:hi]] = data[lo:hi]
    assert dense.mean(axis=0) == pytest.approx([5.3, 0.6, 12.0])


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


def test_per_perturbation_rows_hold_their_means_without_storing_zeros(tmp_path: Path) -> None:
    """A gene averaging under a count is zero in most cells, and those zeros must not be stored.

    Storing them would spend the density cap on nothing and the spec rejects explicit zeros,
    so rows differ in length. The means must still come back exactly.
    """

    output = tmp_path / "pred.h5ad"
    targets = np.array([[10.0, 0.25, 3.0, 0.5], [9.0, 0.5, 3.0, 0.25]], dtype=np.float32)
    write_perturbation_h5ad(
        output,
        gene_names=["g0", "g1", "g2", "g3"],
        perturbations=["p1", "p2"],
        predictions={
            "A": ContextPrediction(
                context="A", gene_indices=np.array([0, 1, 2, 3], dtype=np.int32), means=targets
            )
        },
        cells_per_pert=100,
        cell_patterns=100,
    )

    with h5py.File(output, "r") as handle:
        indptr, indices, data = (handle[f"X/{k}"][()] for k in ("indptr", "indices", "data"))

    assert not (data == 0).any(), "explicit zeros must not be stored"
    assert int(indptr[-1]) < 200 * 4, "a dense write would store every slot"

    dense = np.zeros((200, 4))
    for row in range(200):
        lo, hi = int(indptr[row]), int(indptr[row + 1])
        dense[row, indices[lo:hi]] = data[lo:hi]
    assert dense[:100].mean(axis=0) == pytest.approx(targets[0])
    assert dense[100:].mean(axis=0) == pytest.approx(targets[1])


def test_virtual_storage_does_not_grow_with_the_perturbation_count(tmp_path: Path) -> None:
    """The invariant behind the technique.

    The prediction does not depend on which gene is knocked down, so one context's cells
    serve every perturbation. Stored bytes track the cell count; the logical matrix tracks
    perturbations too. Whether that beats gzip depends on width and compressibility -- at
    1,000 genes per row of real counts it does not -- but this always holds.
    """

    stored = {}
    for n_perturbations in (1, 50):
        path = tmp_path / f"perts_{n_perturbations}.h5ad"
        write_prediction_h5ad(
            path,
            gene_names=["g1", "g2", "g3"],
            perturbations=[f"p{i}" for i in range(n_perturbations)],
            baselines={"A": _baseline()},
            cells_per_pert=4,
        )
        with h5py.File(path, "r") as handle:
            stored[n_perturbations] = handle["uns/_context_cells/A_data"].shape[0]
            assert handle["X/data"].shape[0] == 2 * 4 * n_perturbations

    assert stored[1] == stored[50] == 2 * 4

