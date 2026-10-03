"""Do the K562 effects of a gene's scGPT neighbours predict that gene's effect in another screen?

Held-out check for the neighbour component, built on data this repo already has:

  - reference  : K562 genome-wide screen (every perturbation with enough cells)
  - held-out   : Marson 2025 CD4 T-cell screen, an independent measurement of the same
                 knockdowns in a different cell type, lab and chemistry
  - prediction : for target T, the similarity-weighted K562 effects of the genes nearest T in
                 scGPT space, with T itself excluded from the search

Scored the way the Challenge scores discrimination: is each prediction closest, by cosine, to
its own target's held-out effect among all targets, with every target's own gene removed from
the feature space. 0.5 is chance, and a prediction that is the same for every target (which is
what the generic-response fallback is) scores exactly 0.5.

Controls that must lose for the result to mean anything:
  - shuffled embeddings : same index, same k, genes assigned each other's vectors
  - global mean         : one vector for every target (the generic response's information)

    python scripts/eval_neighbors.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.embeddings import load_gene_embeddings
from vcc2026.io import read_csv_column
from vcc2026.local_eval import _own_gene_columns, perturbation_discrimination
from vcc2026.neighbors import GeneIndex
from vcc2026.transfer import neighbor_fold_changes, secondary_fold_changes, source_effect_table

GRID = [(k, t) for k in (5, 10, 20, 50, 100) for t in (None, 0.1, 0.05, 0.02)]


def pds(prediction: np.ndarray, truth: np.ndarray, own: np.ndarray) -> float:
    return perturbation_discrimination(prediction, truth, own_columns=own)


def mean_pearson(prediction: np.ndarray, truth: np.ndarray) -> float:
    p = prediction - prediction.mean(1, keepdims=True)
    t = truth - truth.mean(1, keepdims=True)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    return float(np.mean(np.where(denom > 0, (p * t).sum(1) / np.where(denom > 0, denom, 1), 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="external/profiles/K562gw_perturbation_means.npz")
    parser.add_argument("--heldout", default="external/profiles/CD4_target_lfc.npz")
    parser.add_argument("--embeddings", default="external/profiles/scgpt_gene_embeddings.npz")
    parser.add_argument("--perts", default="dataset/pert_counts.csv")
    parser.add_argument("--null-draws", type=int, default=50)
    args = parser.parse_args()

    labels, effects, source_genes = source_effect_table(args.source)
    heldout = np.load(args.heldout, allow_pickle=True)
    heldout_genes = [str(g) for g in heldout["genes"]]
    embedded_genes, vectors = load_gene_embeddings(args.embeddings)
    row_of = {g: i for i, g in enumerate(embedded_genes)}
    label_row = {g: i for i, g in enumerate(labels)}

    # A symbol can appear twice on the source axis (TBCE, HSPA14). The pipeline maps both onto
    # one column and the later one wins, so the experiment has to do the same to be comparable.
    held_set = set(heldout_genes)
    shared = list(dict.fromkeys(g for g in source_genes if g in held_set))
    last_column = {g: i for i, g in enumerate(source_genes)}
    src_col = np.array([last_column[g] for g in shared])

    challenge_targets = read_csv_column(args.perts, expected_header="target_gene")
    held_targets = {str(t) for t in heldout["targets"]}
    targets = [
        t for t in challenge_targets if t in held_targets and t in label_row and t in row_of
    ]
    truth, covered, _ = secondary_fold_changes(args.heldout, targets, shared)
    assert covered.all()
    own_fc = effects[[label_row[t] for t in targets]][:, src_col]
    own_cols = _own_gene_columns(targets, shared)
    print(f"{len(targets)} targets measured in both screens, scored over {len(shared):,} shared genes")
    print(f"reference: {len(labels):,} K562 perturbations, "
          f"{sum(l in row_of for l in labels):,} with a scGPT embedding\n")

    reference = [i for i, l in enumerate(labels) if l in row_of]
    ref_names = [labels[i] for i in reference]
    ref_effects = effects[reference][:, src_col]

    def build(vec_of_name) -> GeneIndex:
        return GeneIndex(ref_names, np.stack([vec_of_name(n) for n in ref_names]))

    real_vec = lambda name: vectors[row_of[name]]  # noqa: E731
    index = build(real_vec)
    query_vecs = np.stack([real_vec(t) for t in targets])

    print(f"{'own K562 effect (baseline)':<44}{pds(own_fc, truth, own_cols):>8.4f}"
          f"   r={mean_pearson(own_fc, truth):+.4f}")
    global_mean = np.broadcast_to(ref_effects.mean(0), own_fc.shape)
    print(f"{'global mean (generic-response information)':<44}{pds(global_mean, truth, own_cols):>8.4f}"
          f"   r={mean_pearson(global_mean, truth):+.4f}\n")

    # Shuffled-embedding null: every draw hands each gene another gene's vector, builds one
    # index, and scores the whole grid on it. The mean per configuration is the control column;
    # the per-draw maximum is the null for "best of the grid", which is what gets reported.
    rng = np.random.default_rng(0)
    pool = sorted(set(ref_names) | set(targets))
    null = np.empty((args.null_draws, len(GRID)))
    for draw in range(args.null_draws):
        donor = dict(zip(pool, rng.permutation(pool)))
        shuffled_index = build(lambda n: vectors[row_of[donor[n]]])
        shuffled_q = np.stack([vectors[row_of[donor[t]]] for t in targets])
        for column, (k, temperature) in enumerate(GRID):
            sp = shuffled_index.average_effects(
                ref_effects, targets, shuffled_q, k=k, temperature=temperature
            )
            null[draw, column] = pds(sp, truth, own_cols)

    print("neighbours only (target never retrieves itself)")
    print(f"{'k':>5}{'temperature':>13}{'scGPT PDS':>11}{'r':>9}{'shuffled mean':>15}{'shuffled sd':>13}")
    observed = np.empty(len(GRID))
    for column, (k, temperature) in enumerate(GRID):
        pred = index.average_effects(ref_effects, targets, query_vecs, k=k, temperature=temperature)
        observed[column] = pds(pred, truth, own_cols)
        label = "uniform" if temperature is None else f"{temperature}"
        print(f"{k:>5}{label:>13}{observed[column]:>11.4f}{mean_pearson(pred, truth):>+9.4f}"
              f"{null[:, column].mean():>15.4f}{null[:, column].std():>13.4f}")

    best_column = int(observed.argmax())
    k, temperature = GRID[best_column]
    null_best = null.max(axis=1)
    p_value = float((np.sum(null_best >= observed[best_column]) + 1) / (args.null_draws + 1))
    print(f"\nbest neighbours-only config: k={k}, temperature={temperature}, PDS {observed[best_column]:.4f}")
    print(f"best-of-{len(GRID)} under shuffled embeddings ({args.null_draws} draws): "
          f"mean {null_best.mean():.4f}, sd {null_best.std():.4f}, 95th pct "
          f"{np.percentile(null_best, 95):.4f}, one-sided p = {p_value:.3f}")
    print(f"configs beating their own shuffled mean: {int((observed > null.mean(0)).sum())}/{len(GRID)}")
    nb = index.average_effects(ref_effects, targets, query_vecs, k=k, temperature=temperature)
    print("\nown effect denoised with its neighbours: (1 - w) * own + w * neighbours")
    print(f"{'w':>6}{'PDS':>9}{'r':>9}")
    for w in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0):
        blend = (1 - w) * own_fc + w * nb
        print(f"{w:>6.2f}{pds(blend, truth, own_cols):>9.4f}{mean_pearson(blend, truth):>+9.4f}")

    # The production path must agree with the experiment it was chosen from.
    produced, retrieved, _ = neighbor_fold_changes(
        args.source, args.embeddings, targets, shared, k=k, temperature=temperature
    )
    assert retrieved.all()
    print(f"\nneighbor_fold_changes() reproduces the experiment: "
          f"{np.allclose(produced, nb, atol=1e-5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
