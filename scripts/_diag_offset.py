"""Diagnostic: does the submitted file still encode which perturbation is which?

Builds the same per-perturbation mean profiles the pipeline writes, then scores them in the
official pds_cosine feature space against the *continuous, untruncated* prediction they were
meant to encode. That target is perfectly identifiable by construction, so whatever this
scores is an upper bound on what the file can score against any real reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.baseline import generic_perturbation_response
from vcc2026.io import (
    control_gene_profile,
    read_csv_column,
    read_name_list,
    select_within_budget,
)
from vcc2026.transfer import source_fold_changes

TS = 5e4
ROOT = Path(__file__).resolve().parents[1]


def spec_effect(profiles: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    p = profiles / profiles.sum(axis=1, keepdims=True)
    c = ctrl / ctrl.sum()
    return np.log1p(TS * p) - np.log1p(TS * c)[None, :]


def pds(pred: np.ndarray, true: np.ndarray, drop: np.ndarray) -> float:
    keep = np.ones(pred.shape[1], dtype=bool)
    keep[drop[drop >= 0]] = False
    a, b = pred[:, keep], true[:, keep]
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    d = 1.0 - (a / np.where(an > 0, an, 1)) @ (b / np.where(bn > 0, bn, 1)).T
    d[(an == 0).ravel(), :] = 1.0
    d[:, (bn == 0).ravel()] = 1.0
    own = np.diagonal(d)[:, None]
    closer = (d < own).sum(axis=1)
    tied = (d == own).sum(axis=1) - 1
    n = d.shape[0]
    return float((1.0 - (closer + 0.5 * tied) / (n - 1)).mean())


def quantize(values: np.ndarray, k: int) -> np.ndarray:
    fl = np.floor(values)
    return fl + np.minimum(np.rint((values - fl) * k), k) / k


def main() -> None:
    genes = read_name_list(ROOT / "dataset/gene_names.csv")
    targets = read_csv_column(ROOT / "dataset/pert_counts.csv", expected_header="target_gene")
    print(f"{len(genes)} genes, {len(targets)} targets")

    ctrl, lib, ncells = control_gene_profile(ROOT / "dataset/context_A.h5ad")
    ctrl = ctrl.astype(np.float64)
    print(f"context A: {ncells} control cells, mean library {lib:,.0f} counts")

    lfc, covered = source_fold_changes(
        ROOT / "external/profiles/K562gw_perturbation_means.npz", targets, genes
    )
    fallback = generic_perturbation_response(
        [
            ROOT / "external/profiles/K562_perturbation_means.npz",
            ROOT / "external/profiles/RPE1_perturbation_means.npz",
        ],
        genes,
    )
    lfc = lfc.astype(np.float64)
    missing = ~lfc.any(axis=1)
    lfc[missing] = fallback
    print(f"{covered.sum()}/{len(targets)} covered by gwps; {missing.sum()} on the fallback")

    scale = 0.3
    full = ctrl[None, :] * np.exp(lfc * scale)          # continuous, all 18,533 genes
    drop = np.array([genes.index(t) if t in genes else -1 for t in targets])

    cpm = ctrl * (1e6 / ctrl.sum())
    admitted = cpm >= 5.0
    print(f"{admitted.sum()} genes pass the 5 CPM DE filter; {(ctrl > 0).sum()} nonzero in control")

    variants = {}

    # as submitted (commit 48d497b): top 3,000 by expression, 100 cell patterns
    from vcc2026.io import control_profile_to_sparse

    cols_v1 = control_profile_to_sparse(ctrl, max_genes=3000).gene_indices
    variants["submitted: top3000, K=100"] = (cols_v1, 100, 1.0)

    # current HEAD: whole profile x0.15, cheapest-first over the DE-admitted set, 200 patterns
    scaled = ctrl * 0.15
    cols_v2 = select_within_budget(scaled, entries_per_cell=3000.0, min_mean=1.0 / 200)
    variants["HEAD: x0.15 coverage, K=200"] = (cols_v2, 200, 0.15)

    # what it would be with every control-expressed gene carried
    cols_v3 = np.flatnonzero(ctrl > 0).astype(np.int32)
    variants["all control-expressed, K=200"] = (cols_v3, 200, 0.15)

    ref = spec_effect(full, ctrl)
    s_ref = ref.mean(axis=0)
    print(
        f"\nreference (continuous, untruncated): |shared| = {np.linalg.norm(s_ref):.3f},"
        f" mean |specific| = {np.linalg.norm(ref - s_ref, axis=1).mean():.3f},"
        f" pds vs itself = {pds(ref, ref, drop):.4f}"
    )

    for name, (cols, k, ps) in variants.items():
        sub = np.zeros_like(full)
        sub[:, cols] = quantize(full[:, cols] * ps, k)
        kept = np.count_nonzero(sub.sum(axis=0))
        eff = spec_effect(sub, ctrl)
        shared = eff.mean(axis=0)
        spec = eff - shared
        print(
            f"\n{name}\n"
            f"  genes carried {cols.size:,} ({np.intersect1d(cols, np.flatnonzero(admitted)).size:,}"
            f" of the {admitted.sum():,} DE-scored), surviving quantization {kept:,}\n"
            f"  entries/cell {np.minimum(full[0, cols] * ps, 1).sum():,.0f}\n"
            f"  |shared| = {np.linalg.norm(shared):.3f}   mean |specific| = "
            f"{np.linalg.norm(spec, axis=1).mean():.4f}   ratio = "
            f"{np.linalg.norm(shared) / max(np.linalg.norm(spec, axis=1).mean(), 1e-12):.1f}x\n"
            f"  pds against the profile it encodes = {pds(eff, ref, drop):.4f}\n"
            f"  pds after removing the shared offset = "
            f"{pds(spec, ref - s_ref, drop):.4f}"
        )

    # isolate the two causes: truncation alone, quantization alone
    for name, (cols, k, ps) in variants.items():
        trunc = np.zeros_like(full)
        trunc[:, cols] = full[:, cols]
        e = spec_effect(trunc, ctrl)
        quant = full.copy()
        quant[:, cols] = quantize(full[:, cols] * ps, k) / max(ps, 1e-12)
        eq = spec_effect(quant, ctrl)
        print(
            f"\n{name}: truncation only pds = {pds(e, ref, drop):.4f}; "
            f"quantization only pds = {pds(eq, ref, drop):.4f}"
        )


if __name__ == "__main__":
    main()
