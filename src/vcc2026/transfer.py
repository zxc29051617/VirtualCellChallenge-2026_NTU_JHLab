"""Predict each perturbation separately, by carrying its measured effect over from gwps.

The context-mean baseline says the same thing about all 300 targets, so it cannot do better
than chance at telling them apart -- the discrimination metric's floor is exactly that
prediction. This carries each target's own measured fold change over from Replogle's
genome-wide K562 screen, which covers 267 of the 300.

How much survives the move to a different cell line is the open question. Measured between
Replogle's own two lines, the perturbation-specific component transfers at r = 0.024, which
is useless for magnitude and enough for ranking: a model built on it scores 0.82 on
discrimination against a 0.50 floor. The Challenge's contexts are further away than RPE1 is
from K562, so expect less.

See docs/findings.md sections 1, 7 and 8.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .io import control_gene_profile, control_profile_to_sparse, select_within_budget
from .local_eval import fold_change, load_profiles
from .submission import ContextPrediction


def source_fold_changes(
    profile_path: str | Path,
    targets: list[str],
    gene_names: list[str],
    *,
    control_label: str = "control",
    min_cells: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-target log fold change from a public screen, on `gene_names`' axis.

    Returns (fold_changes[len(targets), len(gene_names)], covered[len(targets)]). Targets the
    screen did not perturb, and genes it did not measure, are left at zero -- those are the
    parts it cannot speak to, and a caller may fill them some other way.
    """

    profiles = load_profiles(profile_path)
    available = {
        label: i
        for i, label in enumerate(profiles.perturbations)
        if profiles.counts[i] >= min_cells
    }
    control = profiles.means[profiles.index_of(control_label)]

    covered = np.array([t in available and t != control_label for t in targets])
    rows = [available[t] for t in targets if t in available and t != control_label]
    measured = fold_change(profiles.means[rows], control)

    position = {gene: i for i, gene in enumerate(gene_names)}
    columns = np.array([position.get(g, -1) for g in profiles.genes])
    known = columns >= 0

    out = np.zeros((len(targets), len(gene_names)), dtype=np.float32)
    out[np.ix_(np.flatnonzero(covered), columns[known])] = measured[:, known]
    return out, covered


def build_transfer_prediction(
    control_path: str | Path,
    *,
    targets: list[str],
    gene_names: list[str],
    target_fold_changes: np.ndarray,
    fallback: np.ndarray | None = None,
    scale: float = 0.3,
    entries_per_cell: float = 3000.0,
    profile_scale: float = 0.15,
    cell_patterns: int = 200,
    row_chunk_size: int = 1024,
) -> ContextPrediction:
    """One context's per-perturbation target means: its control moved by each target's effect.

    `scale` shrinks the carried effect. Least squares put it at 0.27 for K562 to RPE1, and
    shrinking a noisy predictor is right for squared error -- but it is also what collapsed
    discrimination when applied per component (findings section 3), so this scales the whole
    effect uniformly, which cosine distance ignores. There is no way to fit it against the
    Challenge, so it stays a stated assumption.
    """

    mean_counts, _, _ = control_gene_profile(control_path, row_chunk_size=row_chunk_size)

    # Every metric normalizes -- mse reads log(1 + TS * P / sum P), the DE metrics read CPM
    # -- so multiplying the whole profile by a constant changes no score. Density is not
    # invariant: a gene is nonzero in min(mean, 1) of its cells, so scaling down makes every
    # gene averaging under a count proportionally cheaper. That is what makes it affordable
    # to carry all 9,923 genes the DE filter admits rather than the 3,000 richest, and
    # carrying them is worth far more than the mass: a gene left out is read as a predicted
    # expression of zero, not as no claim. See docs/findings.md section 9.
    mean_counts = mean_counts * profile_scale
    columns = select_within_budget(
        mean_counts, entries_per_cell=entries_per_cell, min_mean=1.0 / cell_patterns
    )
    selected = control_profile_to_sparse(mean_counts, columns=columns)

    shifts = target_fold_changes[:, columns] * scale
    if fallback is not None:
        missing = ~target_fold_changes[:, columns].any(axis=1)
        shifts[missing] = fallback[columns]

    with h5py.File(control_path, "r") as handle:
        context = handle["obs/context/categories"][0]
    context = context.decode("utf-8") if isinstance(context, bytes) else str(context)

    return ContextPrediction(
        context=context,
        gene_indices=columns,
        means=(selected.gene_values[None, :] * np.exp(shifts)).astype(np.float32),
    )
