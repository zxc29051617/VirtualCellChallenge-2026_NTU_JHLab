from pathlib import Path

import numpy as np
import pytest

from vcc2026.transfer import blend_fold_changes, blend_secondary_sources, secondary_fold_changes

GENES = ["A", "B", "C", "D"]
TARGETS = ["T1", "T2", "T3"]


def _screen(path: Path, rows: dict[str, list[float]], genes: list[str]) -> Path:
    np.savez_compressed(
        path,
        log_fc=np.array(list(rows.values()), dtype=np.float32),
        targets=np.asarray(list(rows)),
        conditions=np.asarray(["x"] * len(rows)),
        genes=np.asarray(genes),
    )
    return path


def _primary():
    fold = np.array([[1.0, -1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    return fold, np.array([True, True, False])


def test_one_source_matches_the_single_secondary_path_exactly(tmp_path: Path) -> None:
    """v5 used one secondary source; adding a list must not change what it builds."""
    path = _screen(tmp_path / "s.npz", {"T1": [3.0, -3.0], "T3": [4.0, 4.0]}, ["A", "B"])
    fold, covered = _primary()

    second, second_covered, measured = secondary_fold_changes(path, TARGETS, GENES)
    expected, expected_covered, expected_scale = blend_fold_changes(
        fold, covered, second, second_covered, measured, weight=0.1
    )
    got, got_covered, reports = blend_secondary_sources(fold, covered, [(path, 0.1)], TARGETS, GENES)

    assert got.tolist() == expected.tolist()
    assert got_covered.tolist() == expected_covered.tolist()
    assert reports[0]["scale"] == pytest.approx(expected_scale)


def test_a_later_source_can_cover_a_target_no_earlier_one_did(tmp_path: Path) -> None:
    first = _screen(tmp_path / "a.npz", {"T1": [1.0, -1.0]}, ["A", "B"])
    second = _screen(tmp_path / "b.npz", {"T1": [2.0, -2.0], "T3": [2.0, 2.0]}, ["A", "B"])
    fold, covered = _primary()

    out, out_covered, reports = blend_secondary_sources(
        fold, covered, [(first, 0.1), (second, 0.1)], TARGETS, GENES
    )

    assert out_covered.tolist() == [True, True, True]
    assert [r["new_targets"] for r in reports] == [0, 1]
    assert np.abs(out[2, :2]).sum() > 0
    assert out[2, 2:].tolist() == [0.0, 0.0]  # genes the later screen never saw stay unopined


def test_each_source_is_averaged_into_the_running_blend_at_its_own_weight(tmp_path: Path) -> None:
    a = _screen(tmp_path / "a.npz", {"T1": [1.0, -1.0]}, ["A", "B"])
    b = _screen(tmp_path / "b.npz", {"T1": [-1.0, 1.0]}, ["A", "B"])
    fold, covered = _primary()

    only_a, _, _ = blend_secondary_sources(fold, covered, [(a, 0.5)], TARGETS, GENES)
    both, _, _ = blend_secondary_sources(fold, covered, [(a, 0.5), (b, 0.5)], TARGETS, GENES)

    assert only_a[0, :2].tolist() == pytest.approx([1.0, -1.0])  # a agrees with the primary
    # b disagrees, so averaging it in at 0.5 must pull the result toward zero
    assert abs(both[0, 0]) < abs(only_a[0, 0])
    assert both[0, 0] == pytest.approx(0.0)


def test_a_source_sharing_no_target_is_refused_rather_than_filled_with_zeros(tmp_path: Path) -> None:
    disjoint = _screen(tmp_path / "d.npz", {"T3": [2.0, 2.0]}, ["A", "B"])
    fold, covered = _primary()
    with pytest.raises(ValueError, match="shares no target"):
        blend_secondary_sources(fold, covered, [(disjoint, 0.1)], TARGETS, GENES)


def test_no_sources_changes_nothing() -> None:
    fold, covered = _primary()
    out, out_covered, reports = blend_secondary_sources(fold, covered, [], TARGETS, GENES)
    assert out.tolist() == fold.tolist()
    assert out_covered.tolist() == covered.tolist()
    assert reports == []
