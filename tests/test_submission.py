from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from vcc2026.baseline import ContextBaseline
from vcc2026.submission import write_prediction_h5ad


def test_write_prediction_h5ad_creates_readable_h5ad(tmp_path: Path) -> None:
    output = tmp_path / "pred.h5ad"
    baseline = ContextBaseline(
        context="A",
        profile=sparse.csr_matrix(([1.0, 2.0], ([0, 0], [0, 2])), shape=(1, 3)),
        n_control_cells=2,
        mean_library_size=3.0,
    )
    write_prediction_h5ad(
        output,
        gene_names=["g1", "g2", "g3"],
        perturbations=["p1", "p2"],
        baselines={"A": baseline},
        cells_per_pert=2,
        block_rows=1,
    )
    with h5py.File(output, "r") as handle:
        assert tuple(int(x) for x in handle["X"].attrs["shape"]) == (4, 3)
        assert handle["obs/context/codes"].shape == (4,)
        assert handle["obs/target_gene/codes"].shape == (4,)
        assert handle["X/data"].shape[0] == 8
        assert handle["X/indptr"].shape[0] == 5

