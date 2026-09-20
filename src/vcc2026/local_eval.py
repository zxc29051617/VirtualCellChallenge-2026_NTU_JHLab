"""Local scoring against external perturbation data.

The Challenge withholds all ground truth, so nothing can be scored against its own
data. This module scores a prediction against a cell line from Replogle 2022 instead:
fit on one line, predict another whose perturbation responses were never seen.

Fidelity to the official suite, which matters when reading the numbers:

- `pds` follows the official definition closely, including removal of every panel
  target gene from the feature space.
- `expr_error` omits the official delete-one-cell jackknife correction, which needs
  cell-level data. Its no-skill point of 1.0 still holds.
- `direction` and `jaccard` substitute a magnitude gate for the official Wilcoxon
  significance test, which also needs cell-level data. They track the right thing;
  they are not the same statistic.

So local values indicate direction of travel. They are not leaderboard values.
Every metric excludes each perturbation's own target gene, as the official suite does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PDS_TOTAL = 5e4
CPM_TOTAL = 1e6
LFC_EPSILON = 1e-9
LOW_EXPRESSION_CPM = 5.0


@dataclass(frozen=True)
class Profiles:
    """Per-perturbation mean expression, as written by scripts/build_external_profiles.py."""

    means: np.ndarray
    perturbations: list[str]
    counts: np.ndarray
    genes: list[str]

    def index_of(self, perturbation: str) -> int:
        return self.perturbations.index(perturbation)


def load_profiles(path: str | Path) -> Profiles:
    payload = np.load(Path(path), allow_pickle=True)
    return Profiles(
        means=payload["means"],
        perturbations=[str(p) for p in payload["perturbations"]],
        counts=payload["counts"],
        genes=[str(g) for g in payload["genes"]],
    )


def _rescale(means: np.ndarray, total: float) -> np.ndarray:
    row_totals = means.sum(axis=1, keepdims=True)
    return means * (total / np.where(row_totals > 0, row_totals, 1.0))


def log_profiles(means: np.ndarray) -> np.ndarray:
    """Normalize to a fixed total and log-transform, as the discrimination metric does."""
    return np.log1p(_rescale(means, PDS_TOTAL))


def log2_fold_change(means: np.ndarray, control: np.ndarray) -> np.ndarray:
    """log2 fold change of CPM-normalized means against the control profile."""
    pert_cpm = _rescale(means, CPM_TOTAL)
    ctrl_cpm = _rescale(control.reshape(1, -1), CPM_TOTAL)
    return np.log2((pert_cpm + LFC_EPSILON) / (ctrl_cpm + LFC_EPSILON))


def _own_gene_columns(perturbations: list[str], genes: list[str]) -> np.ndarray:
    """Column index of each perturbation's own target gene, or -1 when not measured."""
    lookup = {gene: idx for idx, gene in enumerate(genes)}
    return np.array([lookup.get(p, -1) for p in perturbations], dtype=np.int64)


def _blank_own_gene(matrix: np.ndarray, own_columns: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    rows = np.flatnonzero(own_columns >= 0)
    out[rows, own_columns[rows]] = 0.0
    return out


def perturbation_discrimination(
    predicted: np.ndarray,
    measured: np.ndarray,
    *,
    own_columns: np.ndarray,
) -> float:
    """Mean PDS over perturbations: is each prediction closest to its own measured effect?

    The whole panel's target genes are dropped from the feature space, not one at a time.
    Dropping only the row's own gene would leave every other row's knockdown visible, and a
    submission could win ranks on the published target list alone.
    """

    keep = np.ones(predicted.shape[1], dtype=bool)
    keep[own_columns[own_columns >= 0]] = False
    pred = predicted[:, keep]
    true = measured[:, keep]

    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)
    true_norm = np.linalg.norm(true, axis=1, keepdims=True)
    distances = 1.0 - (pred / np.where(pred_norm > 0, pred_norm, 1.0)) @ (
        true / np.where(true_norm > 0, true_norm, 1.0)
    ).T
    # A zero-norm effect carries no direction, so it ties with everything at distance 1.
    distances[(pred_norm == 0).ravel(), :] = 1.0
    distances[:, (true_norm == 0).ravel()] = 1.0

    n = distances.shape[0]
    own = np.diagonal(distances)[:, None]
    closer = (distances < own).sum(axis=1)
    tied = (distances == own).sum(axis=1) - 1
    return float((1.0 - (closer + 0.5 * tied) / (n - 1)).mean())


def expression_error(
    predicted: np.ndarray,
    measured: np.ndarray,
    *,
    own_columns: np.ndarray,
) -> float:
    """Squared error of predicted effects over squared size of measured effects.

    Predicting no effect at all scores 1.0, matching the official metric's no-skill point.
    Lower is better; 0 is exact.
    """

    residual = _blank_own_gene(predicted - measured, own_columns)
    reference = _blank_own_gene(measured, own_columns)
    return float((residual**2).sum() / (reference**2).sum())


def _gate(values: np.ndarray, top_n: int) -> np.ndarray:
    """Boolean mask of the top_n entries per row by absolute value."""
    if top_n >= values.shape[1]:
        return np.ones_like(values, dtype=bool)
    cut = np.partition(np.abs(values), -top_n, axis=1)[:, -top_n][:, None]
    return np.abs(values) >= cut


def direction_agreement(
    predicted_lfc: np.ndarray,
    measured_lfc: np.ndarray,
    *,
    own_columns: np.ndarray,
    expressed: np.ndarray,
    top_n: int,
) -> float:
    """Fraction of the measured top movers whose predicted direction is correct.

    Chance is 0.5. Proxy for de_wilcoxon_direction_fidelity_yield_raw.
    """

    measured = _blank_own_gene(measured_lfc, own_columns) * expressed
    gate = _gate(measured, top_n)
    agree = np.sign(predicted_lfc) == np.sign(measured)
    hits = (agree & gate).sum(axis=1)
    sizes = gate.sum(axis=1)
    return float(np.mean(hits[sizes > 0] / sizes[sizes > 0]))


def responding_set_jaccard(
    predicted_lfc: np.ndarray,
    measured_lfc: np.ndarray,
    *,
    own_columns: np.ndarray,
    expressed: np.ndarray,
    top_n: int,
) -> float:
    """Jaccard overlap of each side's top movers. Proxy for de_wilcoxon_sig_jaccard."""

    measured = _blank_own_gene(measured_lfc, own_columns) * expressed
    predicted = _blank_own_gene(predicted_lfc, own_columns) * expressed
    true_gate = _gate(measured, top_n)
    pred_gate = _gate(predicted, top_n)
    intersection = (true_gate & pred_gate).sum(axis=1)
    union = (true_gate | pred_gate).sum(axis=1)
    return float(np.mean(intersection[union > 0] / union[union > 0]))


def score(
    predicted_means: np.ndarray,
    measured_means: np.ndarray,
    control_means: np.ndarray,
    *,
    perturbations: list[str],
    genes: list[str],
    top_n: int = 100,
) -> dict[str, float]:
    """Score predicted per-perturbation mean profiles against measured ones."""

    own_columns = _own_gene_columns(perturbations, genes)
    control_log = log_profiles(control_means.reshape(1, -1))

    predicted_effect = log_profiles(predicted_means) - control_log
    measured_effect = log_profiles(measured_means) - control_log

    predicted_lfc = log2_fold_change(predicted_means, control_means)
    measured_lfc = log2_fold_change(measured_means, control_means)
    expressed = _rescale(control_means.reshape(1, -1), CPM_TOTAL) > LOW_EXPRESSION_CPM

    return {
        "pds": perturbation_discrimination(
            predicted_effect, measured_effect, own_columns=own_columns
        ),
        "expr_error": expression_error(
            predicted_effect, measured_effect, own_columns=own_columns
        ),
        "direction": direction_agreement(
            predicted_lfc,
            measured_lfc,
            own_columns=own_columns,
            expressed=expressed,
            top_n=top_n,
        ),
        "jaccard": responding_set_jaccard(
            predicted_lfc,
            measured_lfc,
            own_columns=own_columns,
            expressed=expressed,
            top_n=top_n,
        ),
    }
