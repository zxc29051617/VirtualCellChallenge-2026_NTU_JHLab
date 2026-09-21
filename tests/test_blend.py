from pathlib import Path

import numpy as np
import pytest

from vcc2026.transfer import blend_fold_changes, secondary_fold_changes

GENES = ["A", "B", "C", "D"]
TARGETS = ["T1", "T2", "T3"]


def _write(path: Path, log_fc: np.ndarray, targets: list[str], genes: list[str]) -> Path:
    np.savez_compressed(
        path,
        log_fc=log_fc,
        targets=np.asarray(targets),
        conditions=np.asarray(["x"] * len(targets)),
        genes=np.asarray(genes),
    )
    return path


def test_repeated_rows_for_one_target_are_averaged(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "s.npz",
        np.array([[1.0, 3.0], [3.0, 5.0]]),
        ["T1", "T1"],
        ["A", "B"],
    )
    fold, covered, _ = secondary_fold_changes(path, TARGETS, GENES)
    assert covered.tolist() == [True, False, False]
    assert fold[0].tolist() == [2.0, 4.0, 0.0, 0.0]


def test_genes_the_screen_never_saw_are_not_marked_measured(tmp_path: Path) -> None:
    """The mask is the whole point: a zero here means 'no opinion', not 'no effect'."""
    path = _write(tmp_path / "s.npz", np.array([[1.0, 2.0]]), ["T1"], ["A", "ZZZ"])
    _, _, measured = secondary_fold_changes(path, TARGETS, GENES)
    assert measured.tolist() == [True, False, False, False]


def test_an_unmeasured_gene_keeps_the_primary_value_untouched() -> None:
    primary = np.array([[2.0, 2.0, 2.0, 2.0]])
    secondary = np.zeros((1, 4))
    measured = np.array([True, False, False, False])
    blended, _, _ = blend_fold_changes(
        primary, np.array([True]), secondary, np.array([True]), measured, weight=0.5
    )
    assert blended[0, 1:].tolist() == [2.0, 2.0, 2.0]


def test_the_second_source_is_rescaled_to_the_first_effect_size() -> None:
    """A tenfold difference in units must not become a tenfold difference in influence."""
    primary = np.array([[1.0, -1.0]])
    secondary = np.array([[10.0, -10.0]])
    measured = np.array([True, True])
    blended, _, scale = blend_fold_changes(
        primary, np.array([True]), secondary, np.array([True]), measured, weight=0.5
    )
    assert scale == pytest.approx(0.1)
    assert blended[0].tolist() == pytest.approx([1.0, -1.0])


def test_weight_zero_returns_the_primary_unchanged() -> None:
    primary = np.array([[1.0, 2.0]])
    secondary = np.array([[-5.0, 7.0]])
    measured = np.array([True, True])
    blended, _, _ = blend_fold_changes(
        primary, np.array([True]), secondary, np.array([True]), measured, weight=0.0
    )
    assert blended.tolist() == primary.tolist()


def test_a_target_only_the_second_source_measured_becomes_covered() -> None:
    primary = np.array([[1.0, 1.0], [0.0, 0.0]])
    secondary = np.array([[1.0, 1.0], [4.0, 2.0]])
    measured = np.array([True, True])
    blended, covered, scale = blend_fold_changes(
        primary,
        np.array([True, False]),
        secondary,
        np.array([True, True]),
        measured,
        weight=0.1,
    )
    assert covered.tolist() == [True, True]
    assert blended[1].tolist() == pytest.approx([4.0 * scale, 2.0 * scale])


def test_filling_can_be_turned_off() -> None:
    primary = np.zeros((2, 2))
    primary[0] = 1.0
    secondary = np.ones((2, 2))
    blended, covered, _ = blend_fold_changes(
        primary,
        np.array([True, False]),
        secondary,
        np.array([True, True]),
        np.array([True, True]),
        weight=0.1,
        fill_uncovered=False,
    )
    assert covered.tolist() == [True, False]
    assert blended[1].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("weight", [-0.01, 1.5])
def test_a_weight_outside_zero_to_one_is_refused(weight: float) -> None:
    with pytest.raises(ValueError):
        blend_fold_changes(
            np.zeros((1, 2)),
            np.array([True]),
            np.zeros((1, 2)),
            np.array([True]),
            np.array([True, True]),
            weight=weight,
        )
