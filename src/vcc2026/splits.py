from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    validation: list[int]


def random_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
    stratify_cols: Sequence[str] = ("context", "target_gene"),
) -> SplitIndices:
    rng = pd.Series(range(len(frame)))
    groups = frame.groupby(list(stratify_cols), sort=False).indices
    train: list[int] = []
    validation: list[int] = []
    for _, indices in groups.items():
        indices = list(indices)
        n_val = max(1, int(round(len(indices) * validation_fraction)))
        shuffled = pd.Series(indices).sample(frac=1.0, random_state=seed).tolist()
        validation.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return SplitIndices(train=sorted(train), validation=sorted(validation))


def perturbation_holdout_split(
    frame: pd.DataFrame,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    pert_col: str = "target_gene",
) -> SplitIndices:
    rng = pd.Series(frame[pert_col].unique()).sample(frac=1.0, random_state=seed).tolist()
    n_holdout = max(1, int(round(len(rng) * holdout_fraction)))
    holdout = set(rng[:n_holdout])
    validation = frame.index[frame[pert_col].isin(holdout)].tolist()
    train = frame.index[~frame[pert_col].isin(holdout)].tolist()
    return SplitIndices(train=train, validation=validation)


def cell_context_holdout_split(
    frame: pd.DataFrame,
    *,
    held_out_contexts: Iterable[str],
    context_col: str = "context",
) -> SplitIndices:
    held_out = set(held_out_contexts)
    validation = frame.index[frame[context_col].isin(held_out)].tolist()
    train = frame.index[~frame[context_col].isin(held_out)].tolist()
    return SplitIndices(train=train, validation=validation)

