from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from .io import control_gene_profile, control_profile_to_sparse, load_control_gene_names, validate_gene_alignment


@dataclass(frozen=True)
class ContextBaseline:
    context: str
    profile: sparse.csr_matrix
    n_control_cells: int
    mean_library_size: float


def fit_context_mean_baseline(control_path: str | Path, *, row_chunk_size: int = 1024) -> ContextBaseline:
    with h5py.File(control_path, "r") as handle:
        context_values = handle["obs/context/categories"][()]
        context = context_values[0].decode("utf-8") if isinstance(context_values[0], bytes) else str(context_values[0])
        gene_names = load_control_gene_names(control_path)
        observed_genes = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in handle["var/_index/values"][()]]
        mismatch = validate_gene_alignment(gene_names, observed_genes)
        if mismatch:
            raise ValueError(f"gene alignment failed for {control_path}: {'; '.join(mismatch)}")
        rounded_mean, mean_library_size, n_cells = control_gene_profile(control_path, row_chunk_size=row_chunk_size)
        profile = control_profile_to_sparse(rounded_mean, max_genes=1000).as_csr_row(len(gene_names))
        return ContextBaseline(
            context=context,
            profile=profile,
            n_control_cells=n_cells,
            mean_library_size=mean_library_size,
        )
