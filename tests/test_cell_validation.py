import numpy as np
import pytest

from vcc2026.cell_validation import bh, construct_cells, cpm, read_groups, write_cells
from vcc2026.submission import ContextPrediction, cells_from_mean, write_perturbation_h5ad


def test_low_expression_survives_actual_writer(tmp_path):
    means = np.array([[0.004, 0.01, 3.14159]])
    path = tmp_path / "prediction.h5ad"
    write_perturbation_h5ad(path, gene_names=["low", "tiny", "high"], perturbations=["KO"],
        predictions={"A": ContextPrediction("A", np.arange(3), means)})
    actual = read_groups(path, ["low", "tiny", "high"])["KO"]
    assert actual.shape == (400, 3)
    assert (actual[:, 0] > 0).any()
    assert np.max(np.abs(actual.mean(0) - means)) <= 0.5 / 400 + 1e-7
    assert (actual == np.floor(actual)).all()
    with pytest.raises(ValueError, match="order"):
        read_groups(path, ["high", "tiny", "low"])


def test_repeated_pattern_contract_rejected(tmp_path):
    with pytest.raises(ValueError, match="divide"):
        write_perturbation_h5ad(tmp_path / "bad.h5ad", gene_names=["g"],
            perturbations=["p"], predictions={"A": ContextPrediction("A", np.array([0]),
            np.array([[1.]]))}, cells_per_pert=400, cell_patterns=300)
    assert not (tmp_path / "bad.h5ad").exists()


@pytest.mark.parametrize("values", [[np.nan], [-1], [np.inf]])
def test_invalid_counts_rejected(values):
    with pytest.raises(ValueError):
        cells_from_mean(np.array(values), 400, np.random.default_rng(0))


def test_library_normalization_precedes_gene_filter():
    x = np.array([[1., 99.], [1., 9.]])
    assert np.allclose(cpm(x)[:, 0], [10000, 100000])
    assert not np.allclose(cpm(x)[:, :1], cpm(x[:, :1]))
    assert np.allclose(bh(np.array([.01, .04, .03])), [.03, .04, .04])


def test_materialized_cell_roundtrip(tmp_path):
    cells = np.array([[0, 2], [1, 0]], dtype=np.float32)
    path = write_cells(tmp_path / "cells.h5ad", {"p": cells}, ["a", "b"])
    assert np.array_equal(read_groups(path, ["a", "b"])["p"], cells)


@pytest.mark.parametrize("method", ["floorceil", "nb", "conditioned_nb"])
def test_distribution_samples_contract(method):
    mean = np.array([0., .01, 3.14159, 10.])
    cells = construct_cells(mean, mean + mean**2 / 2, 400, np.random.default_rng(0), method)
    assert cells.shape == (400, 4)
    assert np.isfinite(cells).all() and (cells >= 0).all()
    assert (cells == np.floor(cells)).all() and not cells[:, 0].any()
    if method != "nb":
        assert np.max(np.abs(cells.mean(0) - mean)) <= .5 / 400 + 1e-7


def test_transfer_preserves_low_genes_and_checks_post_effect_budget(tmp_path):
    from vcc2026.transfer import build_transfer_prediction

    genes = ["low", "high"]
    cells = np.zeros((400, 2), dtype=np.float32)
    cells[0, 0], cells[:, 1] = 1, 1
    control = write_cells(tmp_path / "ctrl.h5ad", {"control": cells}, genes, "A")
    kwargs = {"targets": ["KO"], "gene_names": genes,
              "target_fold_changes": np.array([[0., 1.]])}
    prediction = build_transfer_prediction(control, **kwargs)
    assert prediction.gene_indices.tolist() == [0, 1]
    assert prediction.means[0, 0] > 0
    with pytest.raises(ValueError, match="budget"):
        build_transfer_prediction(control, entries_per_cell=.5, **kwargs)
    with pytest.raises(ValueError, match="gene order"):
        build_transfer_prediction(control, **{**kwargs, "gene_names": genes[::-1]})
