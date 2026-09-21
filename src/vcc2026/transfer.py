"""Predict each perturbation separately, by carrying its measured effect over from gwps.

The context-mean baseline says the same thing about all 300 targets, so it cannot do better
than chance at telling them apart -- the discrimination metric's floor is exactly that
prediction. This carries each target's own measured fold change over from Replogle's
genome-wide K562 screen, which covers 267 of the 300.

How much survives the move to a different cell line is the open question. Measured between
Replogle's own two lines, the perturbation-specific component transfers at r = 0.024, which
is useless for magnitude and enough for ranking: a model built on it scores 0.82 on
discrimination against a 0.50 floor before serialization. The Challenge's hidden contexts
cannot be assumed to have the same transfer behavior.

See docs/findings.md sections 1, 7 and 8.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .io import control_gene_profile, read_var_names, select_within_budget
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
    entries_per_cell: float | None = None,
    profile_scale: float = 1.0,
    cell_patterns: int = 400,
    coverage: str = "all_expressed",
    row_chunk_size: int = 1024,
) -> ContextPrediction:
    """One context's per-perturbation target means: its control moved by each target's effect.

    `scale` shrinks the carried effect. Least squares put it at 0.27 for K562 to RPE1, and
    shrinking a noisy predictor is right for squared error -- but it is also what collapsed
    discrimination when applied per component (findings section 3). Scaling before nonlinear
    log normalization can also change cosine rankings. There is no way to fit it against
    the Challenge's hidden responses, so it stays a stated assumption.
    """

    mean_counts, _, _ = control_gene_profile(control_path, row_chunk_size=row_chunk_size)
    if read_var_names(control_path) != gene_names:
        raise ValueError("Control gene order must match the prediction gene axis")
    if target_fold_changes.shape != (len(targets), len(gene_names)):
        raise ValueError("Fold-change dimensions do not match targets and genes")
    if profile_scale <= 0 or not np.isfinite(profile_scale) or cell_patterns < 1:
        raise ValueError("Invalid profile scale or pattern count")
    # Scaling before integerization changes zeros, ties and per-cell library sizes.
    # Preserve all expressed genes by default; a DE gate is not a PDS feature gate.
    mean_counts = mean_counts * profile_scale
    if coverage == "all_expressed":
        columns = np.flatnonzero(mean_counts > 0)
    elif coverage == "legacy_budget":
        if entries_per_cell is None:
            raise ValueError("legacy_budget requires an explicit entries_per_cell")
        columns = select_within_budget(
            mean_counts, entries_per_cell=entries_per_cell, min_mean=1.0 / cell_patterns
        )
    else:
        raise ValueError("coverage must be all_expressed or legacy_budget")

    shifts = target_fold_changes[:, columns] * scale
    if fallback is not None:
        missing = ~target_fold_changes.any(axis=1)
        shifts[missing] = fallback[columns]

    means = (mean_counts[None, columns] * np.exp(shifts)).astype(np.float32)
    if not np.isfinite(means).all():
        raise ValueError("Nonfinite target means")
    # Budget is checked after perturbation effects, never by silently dropping genes.
    if coverage == "all_expressed" and entries_per_cell is not None:
        density = np.minimum(np.rint(means * cell_patterns), cell_patterns).sum(1)
        if np.max(density / cell_patterns) > entries_per_cell:
            raise ValueError("Full coverage exceeds budget; choose an explicit experimental policy")

    with h5py.File(control_path, "r") as handle:
        context = handle["obs/context/categories"][0]
    context = context.decode("utf-8") if isinstance(context, bytes) else str(context)

    return ContextPrediction(
        context=context,
        gene_indices=columns,
        means=means,
    )
