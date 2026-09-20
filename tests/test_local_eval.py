import numpy as np

from vcc2026.local_eval import (
    _own_gene_columns,
    expression_error,
    perturbation_discrimination,
    responding_set_jaccard,
    score,
)

NO_OWN_GENE = np.array([-1, -1, -1], dtype=np.int64)

EFFECTS = np.array(
    [
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 3.0, 0.0, -1.0],
        [0.0, 0.0, 4.0, 0.5],
    ]
)


def test_discrimination_is_one_when_every_prediction_matches_its_own() -> None:
    assert perturbation_discrimination(EFFECTS, EFFECTS, own_columns=NO_OWN_GENE) == 1.0


def test_discrimination_is_one_half_for_a_constant_prediction() -> None:
    """The official spec's stated property: one shared distance row, every rank tied."""
    constant = np.broadcast_to(EFFECTS.mean(axis=0), EFFECTS.shape)
    got = perturbation_discrimination(
        np.ascontiguousarray(constant), EFFECTS, own_columns=NO_OWN_GENE
    )
    assert got == 0.5


def test_discrimination_is_one_half_for_a_zero_prediction() -> None:
    """A zero effect has no direction, so it ties with everything at distance 1."""
    got = perturbation_discrimination(
        np.zeros_like(EFFECTS), EFFECTS, own_columns=NO_OWN_GENE
    )
    assert got == 0.5


def test_expression_error_is_one_when_predicting_no_effect() -> None:
    got = expression_error(np.zeros_like(EFFECTS), EFFECTS, own_columns=NO_OWN_GENE)
    assert got == 1.0


def test_expression_error_is_zero_for_an_exact_prediction() -> None:
    assert expression_error(EFFECTS, EFFECTS, own_columns=NO_OWN_GENE) == 0.0


def test_own_target_gene_is_excluded_from_expression_error() -> None:
    """A prediction wrong only on each row's own target gene must score as if exact."""
    own = np.array([0, 1, 2], dtype=np.int64)
    wrong_on_own_gene = EFFECTS.copy()
    wrong_on_own_gene[[0, 1, 2], [0, 1, 2]] = -99.0
    assert expression_error(wrong_on_own_gene, EFFECTS, own_columns=own) == 0.0


def test_own_gene_columns_maps_symbols_and_marks_unmeasured() -> None:
    got = _own_gene_columns(["B", "MISSING", "A"], ["A", "B"])
    assert got.tolist() == [1, -1, 0]


def test_jaccard_is_one_for_an_exact_prediction() -> None:
    expressed = np.ones((1, EFFECTS.shape[1]), dtype=bool)
    got = responding_set_jaccard(
        EFFECTS, EFFECTS, own_columns=NO_OWN_GENE, expressed=expressed, top_n=2
    )
    assert got == 1.0


def test_score_reports_all_four_metrics_from_mean_profiles() -> None:
    control = np.array([10.0, 10.0, 10.0, 10.0])
    measured = np.array(
        [
            [40.0, 10.0, 10.0, 10.0],
            [10.0, 40.0, 10.0, 10.0],
            [10.0, 10.0, 40.0, 10.0],
        ]
    )
    got = score(
        measured,
        measured,
        control,
        perturbations=["p1", "p2", "p3"],
        genes=["g1", "g2", "g3", "g4"],
        top_n=2,
    )
    assert set(got) == {"pds", "expr_error", "direction", "jaccard"}
    assert got["pds"] == 1.0
    assert got["expr_error"] == 0.0
