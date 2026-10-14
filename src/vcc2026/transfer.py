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

from .embeddings import load_gene_embeddings
from .io import (
    control_gene_dispersion,
    control_gene_profile,
    read_var_names,
    select_within_budget,
)
from .local_eval import fold_change, load_profiles
from .neighbors import GeneIndex
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


def secondary_fold_changes(
    path: str | Path,
    targets: list[str],
    gene_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-target log fold change from a second screen, as written by fetch_cd4_effects.py.

    Returns (fold_changes, covered[len(targets)], measured[len(gene_names)]). `measured` is
    which genes the screen has an opinion about at all, and the caller must keep it: a gene
    it never looked at reads as zero here, and blending a zero in would shrink a real effect
    toward nothing rather than leave it alone.

    Rows repeat per target -- one knockdown assayed in several culture conditions -- and are
    averaged, which is the same denoising argument one level down.
    """

    payload = np.load(Path(path), allow_pickle=True)
    log_fc = payload["log_fc"]
    labels = [str(t) for t in payload["targets"]]
    genes = [str(g) for g in payload["genes"]]

    rows: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        rows.setdefault(label, []).append(index)
    effects = {
        label: np.nan_to_num(np.nanmean(log_fc[index, :], axis=0))
        for label, index in rows.items()
    }

    position = {gene: i for i, gene in enumerate(gene_names)}
    columns = np.array([position.get(g, -1) for g in genes])
    known = columns >= 0

    measured = np.zeros(len(gene_names), dtype=bool)
    measured[columns[known]] = True
    covered = np.array([t in effects for t in targets])
    out = np.zeros((len(targets), len(gene_names)), dtype=np.float32)
    for row, target in enumerate(targets):
        if covered[row]:
            out[row, columns[known]] = effects[target][known]
    return out, covered, measured


def source_effect_table(
    profile_path: str | Path,
    *,
    control_label: str = "control",
    min_cells: int = 25,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Fold change of every well-measured perturbation in a screen, on the screen's own axis.

    Returns (perturbations, fold_changes[n, len(genes)], genes). Unlike `source_fold_changes`
    this is not limited to the Challenge's targets: the neighbour search needs every
    perturbation the screen measured, because the nearest genes to a target are mostly not
    targets themselves.
    """

    profiles = load_profiles(profile_path)
    control = profiles.means[profiles.index_of(control_label)]
    keep = [
        i
        for i, label in enumerate(profiles.perturbations)
        if label != control_label and profiles.counts[i] >= min_cells
    ]
    return (
        [profiles.perturbations[i] for i in keep],
        fold_change(profiles.means[keep], control),
        profiles.genes,
    )


def neighbor_fold_changes(
    source_path: str | Path,
    embeddings_path: str | Path,
    targets: list[str],
    gene_names: list[str],
    *,
    k: int = 20,
    temperature: float | None = 0.05,
    control_label: str = "control",
    min_cells: int = 25,
    index_factory: str = "Flat",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-target effect borrowed from the genes nearest it in scGPT embedding space.

    Returns (fold_changes[len(targets), len(gene_names)], retrieved[len(targets)],
    measured[len(gene_names)]). `retrieved` is false for a target scGPT has no embedding for.
    `measured` is which genes the source screen has an opinion about, and a caller must keep
    it for the same reason `secondary_fold_changes`' caller does: an unmeasured gene reads as
    zero here, and mixing a zero in would shrink a real effect rather than leave it alone.

    A target never retrieves itself, so a target the screen did measure gets an estimate that
    is independent of its own measurement.
    """

    labels, effects, source_genes = source_effect_table(
        source_path, control_label=control_label, min_cells=min_cells
    )
    embedded_genes, vectors = load_gene_embeddings(embeddings_path)
    row_of = {gene: i for i, gene in enumerate(embedded_genes)}

    reference = [i for i, label in enumerate(labels) if label in row_of]
    index = GeneIndex(
        [labels[i] for i in reference],
        vectors[[row_of[labels[i]] for i in reference]],
        factory=index_factory,
    )

    retrieved = np.array([t in row_of for t in targets])
    rows = np.flatnonzero(retrieved)
    averaged = index.average_effects(
        effects[reference],
        [targets[i] for i in rows],
        vectors[[row_of[targets[i]] for i in rows]],
        k=k,
        temperature=temperature,
    )

    position = {gene: i for i, gene in enumerate(gene_names)}
    columns = np.array([position.get(g, -1) for g in source_genes])
    known = columns >= 0
    out = np.zeros((len(targets), len(gene_names)), dtype=np.float32)
    out[np.ix_(rows, columns[known])] = averaged[:, known]
    measured = np.zeros(len(gene_names), dtype=bool)
    measured[columns[known]] = True
    return out, retrieved, measured


def blend_secondary_sources(
    primary: np.ndarray,
    primary_covered: np.ndarray,
    sources: list[tuple[str | Path, float]],
    targets: list[str],
    gene_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fold in several further screens one after another, each at its own weight.

    Each is put in the units of what has been built so far (`blend_fold_changes` matches
    root-mean-square size), so a screen added later is averaged into the running blend rather
    than into the primary alone. A target only a later screen measured is filled from it, so
    coverage grows with every source. Returns (fold_changes, covered, one report per source).
    """

    fold_changes, covered = primary, primary_covered
    reports = []
    for path, weight in sources:
        second, second_covered, measured = secondary_fold_changes(path, targets, gene_names)
        if not (second_covered & covered).any():
            # Its effect size can only be matched to the blend over targets both measured. With
            # none, blend_fold_changes would fill new targets with zeros without saying so.
            raise ValueError(f"{path} shares no target with the sources before it, so its "
                             "effects cannot be put in their units")
        new = int((second_covered & ~covered).sum())
        fold_changes, covered, scale = blend_fold_changes(
            fold_changes, covered, second, second_covered, measured, weight=weight
        )
        reports.append({
            "path": str(path), "weight": weight, "scale": scale,
            "covered": int(second_covered.sum()), "new_targets": new,
            "genes_measured": int(measured.sum()),
        })
    return fold_changes, covered, reports


def apply_neighbor_effects(
    primary: np.ndarray,
    primary_covered: np.ndarray,
    neighbor: np.ndarray,
    retrieved: np.ndarray,
    measured: np.ndarray,
    *,
    weight: float = 0.0,
    fill: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Use neighbour-borrowed effects where the screens give a target little or nothing.

    Returns (fold_changes, covered, n_filled, n_blended).

    `fill` gives a target no screen measured the effect of its scGPT neighbours, in place of
    the generic response that the caller falls back to otherwise. The generic response is one
    vector for every target, so it carries no information about which target it belongs to;
    neighbours do, weakly (findings section 15). Their effects are put in the measured
    targets' units by matching root-mean-square size, as `blend_fold_changes` does for its
    second source -- an average of many profiles is smaller than any one of them, and the
    caller applies one global `scale` assuming every row is a measurement.

    `weight` mixes neighbours into targets that *were* measured: (1 - w) * own + w * nb.
    Both act only on `measured` genes, since a gene the screen never saw reads as zero and
    averaging a zero in would shrink a real effect toward nothing.
    """

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    if primary.shape != neighbor.shape or measured.shape[0] != primary.shape[1]:
        raise ValueError("Fold-change and gene-mask shapes do not line up")

    out = primary.copy()
    covered = primary_covered.copy()

    blended = np.flatnonzero(primary_covered & retrieved) if weight > 0 else np.array([], dtype=int)
    if blended.size:
        out[np.ix_(blended, measured)] = (
            (1.0 - weight) * primary[np.ix_(blended, measured)]
            + weight * neighbor[np.ix_(blended, measured)]
        )

    filled = np.flatnonzero(~primary_covered & retrieved) if fill else np.array([], dtype=int)
    if filled.size:
        reference = np.flatnonzero(primary_covered)
        measured_rms = float(np.sqrt((primary[np.ix_(reference, measured)] ** 2).mean()))
        values = neighbor[np.ix_(filled, measured)]
        own_rms = float(np.sqrt((values**2).mean()))
        gain = measured_rms / own_rms if own_rms > 0 and measured_rms > 0 else 0.0
        out[np.ix_(filled, measured)] = values * gain
        covered[filled] = True
    return out, covered, int(filled.size), int(blended.size)


def blend_fold_changes(
    primary: np.ndarray,
    primary_covered: np.ndarray,
    secondary: np.ndarray,
    secondary_covered: np.ndarray,
    measured: np.ndarray,
    *,
    weight: float,
    fill_uncovered: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Average two independent measurements of the same knockdown. Returns (fc, covered, scale).

    The second screen is put in the first's units by matching root-mean-square effect size
    over the entries both measured, so a blend is an average and nothing else. A least-squares
    fit would be the better predictor but shrinks hard when two sources agree weakly, and
    shrinkage toward no effect lowers squared error on its own, which would read as a win.

    `weight` is small on purpose. Held out in both directions between Replogle's lines, 0.10
    improves all four local metrics while 0.50 is *worse* than the primary alone on
    discrimination: the second source is the weaker measurement, and weighting it equally
    dilutes the stronger one. See findings section 13.
    """

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    if primary.shape != secondary.shape or measured.shape[0] != primary.shape[1]:
        raise ValueError("Fold-change and gene-mask shapes do not line up")

    both = primary_covered & secondary_covered
    shared = primary[np.ix_(both, measured)]
    other = secondary[np.ix_(both, measured)]
    spread = float(np.sqrt((other**2).mean())) if other.size else 0.0
    scale = float(np.sqrt((shared**2).mean()) / spread) if spread > 0 else 0.0

    out = primary.copy()
    rows = np.flatnonzero(both)
    out[np.ix_(rows, measured)] = (
        (1.0 - weight) * primary[np.ix_(rows, measured)]
        + weight * scale * secondary[np.ix_(rows, measured)]
    )

    covered = primary_covered.copy()
    if fill_uncovered:
        # A target only the second screen perturbed. Its scaled effect carries target-specific
        # signal (0.661 discrimination against the first screen, findings section 13) where
        # the generic-response fallback carries none by construction, being one vector for
        # every target. So it is the better filling even though it is the weaker source.
        only = np.flatnonzero(secondary_covered & ~primary_covered)
        out[np.ix_(only, measured)] = scale * secondary[np.ix_(only, measured)]
        covered[only] = True
    return out, covered, scale


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
    with_dispersion: bool = False,
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

    # Dispersion is read from the control at its own depth, since theta is a property of the
    # gene rather than of the level we predict it at. Streaming it costs a second pass.
    theta = None
    if with_dispersion:
        theta = control_gene_dispersion(control_path, row_chunk_size=row_chunk_size)[columns]

    return ContextPrediction(
        context=context,
        gene_indices=columns,
        means=means,
        theta=theta,
    )
