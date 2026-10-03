"""Can scGPT neighbours fill in a target's effect on genes the screen never measured?

K562's genome-wide screen has an effect for 8,248 genes; the Challenge scores 18,533. Every
other gene reads as a predicted fold change of exactly zero, which the direction metric counts
as a miss. This asks whether a gene's scGPT neighbours can stand in:

  hold out a random fifth of the screen's measured genes; for each Challenge target, predict
  each held-out gene's effect as the similarity-weighted mean of that same target's effects on
  the nearest *kept* genes in scGPT space; compare with what was measured.

Two numbers, both over held-out genes only:
  r    mean per-target Pearson correlation of predicted and measured effects
  PDS  is each target's prediction closest to its own measured profile among all targets,
       with every target's own gene removed (0.5 is chance)

Control: the same search with gene embeddings shuffled. Whatever it scores is what structure
in the effects alone, with no knowledge of which genes are related, can explain.

    python scripts/eval_gene_axis_fill.py
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
from vcc2026.transfer import secondary_fold_changes, source_effect_table


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
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--shuffles", type=int, default=3)
    args = parser.parse_args()

    labels, effects, source_genes = source_effect_table(args.source)
    embedded, vectors = load_gene_embeddings(args.embeddings)
    row_of = {g: i for i, g in enumerate(embedded)}
    label_row = {g: i for i, g in enumerate(labels)}

    targets = [
        t for t in read_csv_column(args.perts, expected_header="target_gene") if t in label_row
    ]
    # One column per symbol (the later duplicate wins, as everywhere else), embedded only.
    last = {g: i for i, g in enumerate(source_genes)}
    genes = [g for g in dict.fromkeys(source_genes) if g in row_of]
    effect = effects[[label_row[t] for t in targets]][:, [last[g] for g in genes]]
    gene_vectors = vectors[[row_of[g] for g in genes]]
    print(f"{len(targets)} Challenge targets measured, {len(genes):,} embedded source genes\n")

    configs = [(5, None), (20, None), (20, 0.05), (50, None)]
    rng = np.random.default_rng(0)
    results = {c: {"real": [], "shuf": []} for c in configs}

    for split in range(args.splits):
        order = rng.permutation(len(genes))
        n_held = int(len(genes) * args.holdout)
        held, kept = np.sort(order[:n_held]), np.sort(order[n_held:])
        held_names = [genes[i] for i in held]
        kept_names = [genes[i] for i in kept]
        own = _own_gene_columns(targets, held_names)
        truth = effect[:, held]
        reference = effect[:, kept].T  # [kept genes, targets]

        def score(index_vecs: np.ndarray, query_vecs: np.ndarray, k: int, temp) -> tuple[float, float]:
            index = GeneIndex(kept_names, index_vecs)
            pred = index.average_effects(
                reference, held_names, query_vecs, k=k, temperature=temp, exclude_self=False
            ).T  # [targets, held genes]
            return (
                mean_pearson(pred, truth),
                perturbation_discrimination(pred, truth, own_columns=own),
            )

        for k, temp in configs:
            results[(k, temp)]["real"].append(
                score(gene_vectors[kept], gene_vectors[held], k, temp)
            )
            for _ in range(args.shuffles):
                perm = rng.permutation(len(genes))
                shuffled = gene_vectors[perm]
                results[(k, temp)]["shuf"].append(score(shuffled[kept], shuffled[held], k, temp))

    print("A. genes held out of the same screen (K562 -> K562)")
    print(f"{'k':>5}{'temperature':>13}{'r (scGPT)':>12}{'r (shuffled)':>14}{'PDS (scGPT)':>13}{'PDS (shuffled)':>16}")
    for (k, temp), res in results.items():
        real, shuf = np.array(res["real"]), np.array(res["shuf"])
        label = "uniform" if temp is None else f"{temp}"
        print(f"{k:>5}{label:>13}{real[:, 0].mean():>12.4f}{shuf[:, 0].mean():>14.4f}"
              f"{real[:, 1].mean():>13.4f}{shuf[:, 1].mean():>16.4f}")

    # B. The production case: genes K562 never measured, scored against another screen that did.
    # The CD4 screen measured ~3,600 genes K562 did not. Predict a target's CD4 effect on each of
    # them from its K562 effects on the nearest genes K562 *did* measure, across cell types.
    heldout = np.load(args.heldout, allow_pickle=True)
    held_axis = [str(g) for g in heldout["genes"]]
    source_set = set(source_genes)
    outside = [g for g in dict.fromkeys(held_axis) if g not in source_set and g in row_of]
    cd4_targets = [t for t in targets if t in {str(x) for x in heldout["targets"]}]
    truth_cd4, covered, _ = secondary_fold_changes(args.heldout, cd4_targets, outside)
    assert covered.all()
    cd4_own = _own_gene_columns(cd4_targets, outside)
    reference = effect[[targets.index(t) for t in cd4_targets]].T  # [source genes, targets]
    print(f"\nB. genes K562 never measured, scored against CD4 ({len(cd4_targets)} targets, "
          f"{len(outside):,} CD4-only genes)")
    print(f"{'k':>5}{'temperature':>13}{'r (scGPT)':>12}{'r (shuffled)':>14}{'PDS (scGPT)':>13}"
          f"{'PDS (shuffled)':>16}{'OLS gain':>10}")
    outside_vecs = vectors[[row_of[g] for g in outside]]
    for k, temp in configs:
        index = GeneIndex(genes, gene_vectors)
        pred = index.average_effects(
            reference, outside, outside_vecs, k=k, temperature=temp, exclude_self=False
        ).T
        real_r = mean_pearson(pred, truth_cd4)
        real_pds = perturbation_discrimination(pred, truth_cd4, own_columns=cd4_own)
        gain = float((pred * truth_cd4).sum() / (pred * pred).sum())
        shuf_r, shuf_pds = [], []
        for _ in range(args.shuffles):
            perm = rng.permutation(len(genes))
            sp = GeneIndex(genes, gene_vectors[perm]).average_effects(
                reference, outside, outside_vecs, k=k, temperature=temp, exclude_self=False
            ).T
            shuf_r.append(mean_pearson(sp, truth_cd4))
            shuf_pds.append(perturbation_discrimination(sp, truth_cd4, own_columns=cd4_own))
        label = "uniform" if temp is None else f"{temp}"
        print(f"{k:>5}{label:>13}{real_r:>12.4f}{np.mean(shuf_r):>14.4f}{real_pds:>13.4f}"
              f"{np.mean(shuf_pds):>16.4f}{gain:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
