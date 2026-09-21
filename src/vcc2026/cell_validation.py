"""Cell-level offline diagnostics; not a replica-normalized leaderboard implementation."""
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse, stats

from .io import read_categorical, read_var_names
from .submission import _write_frame, cells_from_mean, cells_from_nb


def read_rows(path, rows, columns=None):
    """Read requested rows without loading a whole dense H5AD (chunk-aligned scan)."""
    rows = np.asarray(rows, dtype=np.int64)
    with h5py.File(path, "r") as f:
        x = f["X"]
        ncols = x.shape[1] if isinstance(x, h5py.Dataset) else int(x.attrs["shape"][1])
        cols = np.arange(ncols) if columns is None else np.asarray(columns)
        out = np.empty((len(rows), len(cols)), dtype=np.float32)
        if isinstance(x, h5py.Dataset):
            size = x.chunks[0] if x.chunks else 1024
            for block in np.unique(rows // size):
                pick = np.flatnonzero(rows // size == block)
                start = int(block * size)
                out[pick] = x[start:start + size][rows[pick] - start][:, cols]
        else:
            ptr = x["indptr"][:]
            if len(rows) and np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
                lo, hi = int(ptr[rows[0]]), int(ptr[rows[-1] + 1])
                block = sparse.csr_matrix((x["data"][lo:hi], x["indices"][lo:hi],
                    ptr[rows[0]:rows[-1] + 2] - lo), shape=(len(rows), ncols))
                return block[:, cols].toarray()
            for i, row in enumerate(rows):
                lo, hi = int(ptr[row]), int(ptr[row + 1])
                dense = np.zeros(ncols, dtype=np.float32)
                dense[x["indices"][lo:hi]] = x["data"][lo:hi]
                out[i] = dense[cols]
    return out


def write_cells(path, groups, genes, context="local"):
    """Materialized submission-shaped H5AD, preserving the entire named gene axis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(next(iter(groups.values())))
    for cells in groups.values():
        if (cells.shape != (n, len(genes)) or not np.isfinite(cells).all()
                or (cells < 0).any() or (cells != np.floor(cells)).any()):
            raise ValueError("Expected equally sized groups of finite nonnegative integer counts")
    x = sparse.vstack([sparse.csr_matrix(v) for v in groups.values()], format="csr")
    with h5py.File(path, "w") as f:
        f.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        for name in ("layers", "obsm", "obsp", "uns", "varm", "varp"):
            f.create_group(name).attrs.update(
                {"encoding-type": "dict", "encoding-version": "0.1.0"})
        node = f.create_group("X")
        node.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                           "shape": x.shape})
        for name in ("data", "indices", "indptr"):
            node.create_dataset(name, data=getattr(x, name), compression="gzip")
        _write_frame(f, genes, list(groups), [context], n)
    return path


def read_groups(path, expected_genes):
    if read_var_names(path) != expected_genes:
        raise ValueError("Readback gene order mismatch")
    contexts, _ = read_categorical(path, "context")
    if len(np.unique(contexts)) != 1:
        raise ValueError("Select a single context before reading local evaluation groups")
    codes, labels = read_categorical(path, "target_gene")
    return {label: read_rows(path, np.flatnonzero(codes == i)) for i, label in enumerate(labels)}


def cpm(cells):
    """Normalize on the FULL output axis before applying an expression gate."""
    x = np.asarray(cells, dtype=np.float64)
    return x * (1e6 / np.maximum(x.sum(axis=1, keepdims=True), 1))


def bh(p):
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(q, 1)
    return result


def differential_expression(cells, reference, exclude=()):
    """Tie-corrected asymptotic rank-sum, BH over reference-CPM>5 nonexcluded genes.

    Diagnostic convention: no continuity correction. The exact official server's
    implementation must still be checked; these are not official DE scores.
    """
    a, b = cpm(cells), cpm(reference)
    gate = b.mean(0) > 5
    gate[list(exclude)] = False
    cols = np.flatnonzero(gate)
    p = np.ones(len(cols))
    for start in range(0, len(cols), 128):
        c = cols[start:start + 128]
        p[start:start + len(c)] = stats.mannwhitneyu(
            a[:, c], b[:, c], axis=0, method="asymptotic", use_continuity=False).pvalue
    p = np.nan_to_num(p, nan=1.0)
    called = np.zeros(cells.shape[1], dtype=bool)
    called[cols] = bh(p) < 0.05
    return {"called": called, "tested": gate,
            "lfc": np.log2((a.mean(0) + 1e-9) / (b.mean(0) + 1e-9))}


def construct_cells(mean, variance, n, rng, method):
    if method == "floorceil":
        return cells_from_mean(mean, n, rng)
    theta = np.divide(mean**2, variance - mean, out=np.full_like(mean, 1e6),
                      where=variance > mean + 1e-10)
    theta = np.clip(theta, 1e-3, 1e6)
    if method == "nb":
        return rng.negative_binomial(theta, theta / (theta + mean),
                                    size=(n, len(mean))).astype(np.float32)
    if method == "conditioned_nb":
        # The same construction the submission writer uses, so the null diagnostic below
        # measures what actually ships rather than a second implementation of it.
        return cells_from_nb(mean, theta, n, rng)
    raise ValueError(method)
