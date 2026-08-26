import pandas as pd

from vcc2026.splits import cell_context_holdout_split, perturbation_holdout_split, random_split


def test_random_split_produces_both_sets() -> None:
    frame = pd.DataFrame(
        {
            "context": ["A", "A", "B", "B"],
            "target_gene": ["g1", "g1", "g2", "g2"],
        }
    )
    split = random_split(frame, validation_fraction=0.5, seed=0)
    assert split.train
    assert split.validation


def test_perturbation_holdout_leaves_entire_perturbations_out() -> None:
    frame = pd.DataFrame(
        {
            "context": ["A", "A", "A", "A"],
            "target_gene": ["g1", "g1", "g2", "g3"],
        }
    )
    split = perturbation_holdout_split(frame, holdout_fraction=0.5, seed=0)
    assert set(frame.loc[split.validation, "target_gene"]).isdisjoint(set(frame.loc[split.train, "target_gene"]))


def test_cell_context_holdout_targets_exact_contexts() -> None:
    frame = pd.DataFrame(
        {
            "context": ["A", "A", "B", "C"],
            "target_gene": ["g1", "g2", "g1", "g3"],
        }
    )
    split = cell_context_holdout_split(frame, held_out_contexts=["B", "C"])
    assert set(frame.loc[split.validation, "context"]) == {"B", "C"}
    assert set(frame.loc[split.train, "context"]) == {"A"}

