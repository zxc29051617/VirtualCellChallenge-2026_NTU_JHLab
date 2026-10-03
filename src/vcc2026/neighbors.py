"""Nearest-neighbour retrieval over gene embeddings, with FAISS as the index.

A Challenge target is only a gene name. Where no screen measured its knockdown, the effects
of the genes whose scGPT embeddings sit closest to it are the best evidence on offer: genes
in one complex or pathway tend to produce similar knockdown signatures (Replogle 2022
clusters perturbations into complexes on exactly this basis).

Search is exact cosine similarity (`Flat` over unit vectors, inner product), because the
reference set is ~10k genes and an approximate index would only add a recall question to
what is already a noisy signal. `factory` accepts any FAISS index string for a larger set.
"""

from __future__ import annotations

import numpy as np

QUERY_BLOCK = 256


def _unit(vectors: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms > 0, norms, 1.0)


def neighbour_weights(similarities: np.ndarray, temperature: float | None) -> np.ndarray:
    """Softmax over each row's similarities. `None` weights every neighbour equally.

    A small temperature lets the closest gene dominate; a large one approaches the plain
    mean, which is mostly the response every perturbation shares.
    """

    if temperature is None:
        return np.full_like(similarities, 1.0 / similarities.shape[1], dtype=np.float32)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = (similarities - similarities.max(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


class GeneIndex:
    """Exact cosine search over the embeddings of a set of reference genes."""

    def __init__(self, genes: list[str], vectors: np.ndarray, *, factory: str = "Flat") -> None:
        import faiss  # optional dependency

        if len(genes) != len(vectors):
            raise ValueError("One embedding is needed per gene")
        if len(genes) == 0:
            raise ValueError("Cannot index an empty gene set")
        self.genes = list(genes)
        units = _unit(vectors)
        self.index = faiss.index_factory(units.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
        if not self.index.is_trained:
            self.index.train(units)
        self.index.add(units)

    def __len__(self) -> int:
        return len(self.genes)

    def search(
        self,
        query_genes: list[str],
        query_vectors: np.ndarray,
        k: int,
        *,
        exclude_self: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(similarities[n, k], reference rows[n, k]) of each query's nearest reference genes.

        With `exclude_self` a query never retrieves itself, which is what makes a leave-one-out
        measurement honest when the query is also in the reference set.
        """

        if len(query_genes) != len(query_vectors):
            raise ValueError("One query vector is needed per query gene")
        k = min(k, len(self) - (1 if exclude_self else 0))
        if k < 1:
            raise ValueError("The reference set is too small to leave anything to retrieve")
        probe = k + 1 if exclude_self else k
        sims, ids = self.index.search(_unit(query_vectors), probe)

        out_sims = np.empty((len(query_genes), k), dtype=np.float32)
        out_ids = np.empty((len(query_genes), k), dtype=np.int64)
        for row, gene in enumerate(query_genes):
            keep = np.ones(probe, dtype=bool)
            if exclude_self:
                own = [i for i, j in enumerate(ids[row]) if j >= 0 and self.genes[j] == gene]
                keep[own[0] if own else probe - 1] = False
            out_sims[row] = sims[row][keep][:k]
            out_ids[row] = ids[row][keep][:k]
        return out_sims, out_ids

    def average_effects(
        self,
        reference_effects: np.ndarray,
        query_genes: list[str],
        query_vectors: np.ndarray,
        *,
        k: int,
        temperature: float | None,
        exclude_self: bool = True,
    ) -> np.ndarray:
        """Similarity-weighted mean of the nearest reference genes' effect vectors.

        `reference_effects` has one row per indexed gene, in index order.
        """

        if reference_effects.shape[0] != len(self):
            raise ValueError("reference_effects must have one row per indexed gene")
        out = np.empty((len(query_genes), reference_effects.shape[1]), dtype=np.float32)
        for start in range(0, len(query_genes), QUERY_BLOCK):
            block = slice(start, start + QUERY_BLOCK)
            sims, ids = self.search(
                query_genes[block], query_vectors[block], k, exclude_self=exclude_self
            )
            weights = neighbour_weights(sims, temperature)
            out[block] = np.einsum("qk,qkf->qf", weights, reference_effects[ids])
        return out
