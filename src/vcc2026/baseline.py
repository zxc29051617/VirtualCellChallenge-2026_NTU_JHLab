from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from .io import (
    control_gene_profile,
    control_profile_to_sparse,
    load_control_gene_names,
    validate_gene_alignment,
)
from .local_eval import fold_change, load_profiles


def generic_perturbation_response(
    profile_paths: Sequence[str | Path],
    gene_names: list[str],
    *,
    control_label: str = "control",
    min_cells: int = 25,
) -> np.ndarray:
    """Log fold change shared by every perturbation, averaged over public cell lines.

    Breaking any essential gene produces a generic response on top of whatever is specific
    to that gene. Predicting the control unchanged says nothing about it, and the official
    direction metric counts a zero predicted fold change as a miss rather than an abstention
    -- so a control-pasting submission forfeits that metric outright rather than scoring at
    chance. This recovers a direction for the genes public data can speak to.

    Returned on `gene_names`' axis; genes absent from every source keep zero.
    """

    total = np.zeros(len(gene_names), dtype=np.float64)
    seen = np.zeros(len(gene_names), dtype=np.int64)
    position = {gene: i for i, gene in enumerate(gene_names)}

    for path in profile_paths:
        profiles = load_profiles(path)
        control = profiles.means[profiles.index_of(control_label)]
        rows = [
            i
            for i, label in enumerate(profiles.perturbations)
            if label != control_label and profiles.counts[i] >= min_cells
        ]
        shared = fold_change(profiles.means[rows], control).mean(axis=0)

        columns = np.array([position.get(g, -1) for g in profiles.genes])
        known = columns >= 0
        total[columns[known]] += shared[known]
        seen[columns[known]] += 1

    return np.divide(total, seen, out=np.zeros_like(total), where=seen > 0)


@dataclass(frozen=True)
class ContextBaseline:
    context: str
    profile: sparse.csr_matrix
    n_control_cells: int
    mean_library_size: float


# What bounds the gene count is `vcc prep`, not the scoring cap. It loads the whole indices
# array as int64, so a submission costs about 12 bytes per stored entry while being packaged:
# all 7,608 genes that survive rounding would ask for 33 GB and fail on a 31 GB machine, well
# inside the 4.75e9 entries the scorer allows. 3,000 keeps 72% of the control's expression at
# about 13 GB. Raise it on a larger machine; the cost of lowering it is signal, not format.
DEFAULT_MAX_GENES = 3000


def fit_context_mean_baseline(
    control_path: str | Path,
    *,
    row_chunk_size: int = 1024,
    max_genes: int | None = DEFAULT_MAX_GENES,
    response_shift: np.ndarray | None = None,
    response_scale: float = 1.0,
) -> ContextBaseline:
    with h5py.File(control_path, "r") as handle:
        context_values = handle["obs/context/categories"][()]
        context = context_values[0].decode("utf-8") if isinstance(context_values[0], bytes) else str(context_values[0])
        gene_names = load_control_gene_names(control_path)
        observed_genes = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in handle["var/_index/values"][()]]
        mismatch = validate_gene_alignment(gene_names, observed_genes)
        if mismatch:
            raise ValueError(f"gene alignment failed for {control_path}: {'; '.join(mismatch)}")
        mean_counts, mean_library_size, n_cells = control_gene_profile(
            control_path, row_chunk_size=row_chunk_size
        )
        if response_shift is not None:
            # Applied before rounding: a shift too small to move a gene's integer count
            # leaves its predicted fold change at zero, which the direction metric reads
            # as a miss rather than as no claim.
            mean_counts = mean_counts * np.exp(response_scale * response_shift)
        profile = control_profile_to_sparse(mean_counts, max_genes=max_genes).as_csr_row(
            len(gene_names)
        )
        return ContextBaseline(
            context=context,
            profile=profile,
            n_control_cells=n_cells,
            mean_library_size=mean_library_size,
        )
