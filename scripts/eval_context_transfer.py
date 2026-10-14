"""Does scGPT's context-dependent view of a gene predict how well its effects transfer?

Outcome, per gene g: a_g = correlation, across the perturbations both K562 and RPE1 measured,
between g's log fold change in K562 and in RPE1 (each perturbation's own target excluded for
its own gene). It is the gene-level version of "does the K562 effect predict the RPE1 one".
Thousands of genes, so this has power that a comparison of two or three contexts cannot.

Feature, per gene: how similar scGPT's transformer state for g is in K562 control cells and in
RPE1 control cells (cosine after removing each context's average gene state, which otherwise
swamps it). The part that is specific to the *pair* of contexts is that similarity minus the
similarity between two samples of the same context.

The fair question is whether the feature adds anything beyond what is already known about a
gene without scGPT: how highly it is expressed in each line and how much it moves in each. So
the score is the gain in cross-validated R^2 from adding it to those covariates, against a null
that shuffles the feature across genes.

    python scripts/eval_context_transfer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vcc2026.transfer import source_effect_table


def centered_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.where(den > 0, den, 1.0)


def gene_states(path: Path, genes: list[str], min_count: float) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    index = {g: i for i, g in enumerate(payload["genes"])}
    keep = [g for g in genes if g in index]
    rows = np.array([index[g] for g in keep])
    out = {"genes": np.array(keep)}
    ok = np.ones(len(keep), dtype=bool)
    for half in (0, 1):
        count = payload[f"gene_count_{half}"][rows]
        out[f"state_{half}"] = payload[f"gene_sum_{half}"][rows] / np.maximum(count, 1)[:, None]
        ok &= count >= min_count
    out["ok"] = ok
    symbol_index = {g: i for i, g in enumerate(payload["symbols"])}
    out["mean_counts"] = np.array([payload["mean_counts"][symbol_index[g]] for g in keep])
    return out


def ols_r2(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
    resid = test_y - test_x @ beta
    return 1.0 - float((resid**2).sum() / ((test_y - test_y.mean()) ** 2).sum())


def cv_r2(x: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    scores = []
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        scores.append(ols_r2(x[train], y[train], x[test], y[test]))
    return float(np.mean(scores))


def design(columns: list[np.ndarray]) -> np.ndarray:
    z = [(c - c.mean()) / (c.std() + 1e-12) for c in columns]
    return np.column_stack([np.ones(len(z[0]))] + z + [c**2 for c in z])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="external/profiles")
    parser.add_argument("--states", default="external/profiles/scgpt_context")
    parser.add_argument("--min-count", type=float, default=15.0, help="min cells per half a gene state rests on")
    parser.add_argument("--perms", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    k_labels, k_fc, k_genes = source_effect_table(f"{args.profiles}/K562_perturbation_means.npz")
    r_labels, r_fc, r_genes = source_effect_table(f"{args.profiles}/RPE1_perturbation_means.npz")
    k_col = {g: i for i, g in enumerate(k_genes)}
    r_col = {g: i for i, g in enumerate(r_genes)}
    shared_perts = [p for p in k_labels if p in set(r_labels)]
    shared_genes = [g for g in dict.fromkeys(k_genes) if g in r_col]
    kp = {p: i for i, p in enumerate(k_labels)}
    rp = {p: i for i, p in enumerate(r_labels)}
    x = k_fc[[kp[p] for p in shared_perts]][:, [k_col[g] for g in shared_genes]].astype(np.float64)
    y = r_fc[[rp[p] for p in shared_perts]][:, [r_col[g] for g in shared_genes]].astype(np.float64)
    print(f"{len(shared_perts):,} shared perturbations, {len(shared_genes):,} shared genes")

    weight = np.ones_like(x)
    pert_row = {p: i for i, p in enumerate(shared_perts)}
    for j, g in enumerate(shared_genes):
        if g in pert_row:
            weight[pert_row[g], j] = 0.0  # a gene's own knockdown is not a response
    n = weight.sum(0)
    mx, my = (weight * x).sum(0) / n, (weight * y).sum(0) / n
    cov = (weight * (x - mx) * (y - my)).sum(0)
    sx = np.sqrt((weight * (x - mx) ** 2).sum(0) / n)
    sy = np.sqrt((weight * (y - my) ** 2).sum(0) / n)
    agreement = cov / n / np.maximum(sx * sy, 1e-12)

    k = gene_states(Path(args.states) / "K562.npz", shared_genes, args.min_count)
    r = gene_states(Path(args.states) / "RPE1.npz", shared_genes, args.min_count)
    both = [g for g in shared_genes if g in set(k["genes"]) and g in set(r["genes"])]
    ki = {g: i for i, g in enumerate(k["genes"])}
    ri = {g: i for i, g in enumerate(r["genes"])}
    keep = np.array([k["ok"][ki[g]] and r["ok"][ri[g]] for g in both])
    genes = [g for g, ok in zip(both, keep) if ok]
    rows_k = np.array([ki[g] for g in genes])
    rows_r = np.array([ri[g] for g in genes])
    sel = np.array([shared_genes.index(g) for g in genes])

    ks = [k[f"state_{h}"][rows_k] for h in (0, 1)]
    rs = [r[f"state_{h}"][rows_r] for h in (0, 1)]
    cross = np.mean([centered_cosine(a, b) for a in ks for b in rs], axis=0)
    within = 0.5 * (centered_cosine(ks[0], ks[1]) + centered_cosine(rs[0], rs[1]))
    shift = cross - within

    a = agreement[sel]
    covariates = [
        np.log1p(k["mean_counts"][rows_k]),
        np.log1p(r["mean_counts"][rows_r]),
        sx[sel],
        sy[sel],
    ]
    print(f"{len(genes):,} genes with >= {args.min_count:g} cells behind each of four gene states\n")
    print(f"agreement a_g: mean {a.mean():+.3f}, sd {a.std():.3f}")

    def spearman(u: np.ndarray, v: np.ndarray) -> float:
        ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v))
        return float(np.corrcoef(ru, rv)[0, 1])

    print("\nrank correlation of each quantity with a_g:")
    for name, col in [("log mean counts, K562", covariates[0]), ("log mean counts, RPE1", covariates[1]),
                      ("effect sd, K562", covariates[2]), ("effect sd, RPE1", covariates[3]),
                      ("scGPT cross-context similarity", cross),
                      ("scGPT same-context similarity (noise)", within),
                      ("scGPT context-specific shift", shift)]:
        print(f"  {name:<42}{spearman(col, a):>+8.3f}")

    rng = np.random.default_rng(0)
    base = design(covariates)
    rows = []
    for rep in range(args.repeats):
        order = rng.permutation(len(a))
        folds = [order[i :: 5] for i in range(5)]
        rows.append(
            (cv_r2(base, a, folds), cv_r2(design(covariates + [cross]), a, folds),
             cv_r2(design(covariates + [cross, within]), a, folds))
        )
    r2_base, r2_cross, r2_both = np.mean(rows, axis=0)
    print(f"\ncross-validated R^2 for a_g (5-fold over genes, {args.repeats} repeats)")
    print(f"  covariates only (expression, effect size)          {r2_base:.4f}")
    print(f"  + scGPT cross-context similarity                   {r2_cross:.4f}   gain {r2_cross - r2_base:+.4f}")
    print(f"  + cross-context and same-context similarity        {r2_both:.4f}   gain {r2_both - r2_base:+.4f}")

    order = rng.permutation(len(a))
    folds = [order[i :: 5] for i in range(5)]
    observed = cv_r2(design(covariates + [cross, within]), a, folds) - cv_r2(base, a, folds)
    null = []
    for _ in range(args.perms):
        p = rng.permutation(len(a))
        null.append(cv_r2(design(covariates + [cross[p], within[p]]), a, folds) - cv_r2(base, a, folds))
    null = np.array(null)
    print(f"\nshuffling the scGPT features across genes ({args.perms} draws): gain mean {null.mean():+.4f}, "
          f"sd {null.std():.4f}, observed {observed:+.4f}, one-sided p = "
          f"{(np.sum(null >= observed) + 1) / (args.perms + 1):.3f}")

    # Does weighting each gene's transferred effect by its predicted transferability help?
    # Weights come from a model that never saw the gene (5-fold over genes), and use only what a
    # new Challenge context would have: both contexts' expression, the source's effect size, and
    # scGPT's states. The target context's effect size is NOT used, since it would be unknown.
    from vcc2026.local_eval import _own_gene_columns, perturbation_discrimination

    xs, ys = x[:, sel], y[:, sel]
    own = _own_gene_columns(shared_perts, genes)
    available = [covariates[0], covariates[1], covariates[2]]
    feature_sets = {
        "expression + source effect size": design(available),
        "  + scGPT states": design(available + [cross, within]),
    }
    order = rng.permutation(len(a))
    folds = [order[i :: 5] for i in range(5)]

    def held_out_weights(features: np.ndarray) -> np.ndarray:
        out = np.empty(len(a))
        for test in folds:
            train = np.setdiff1d(np.arange(len(a)), test)
            beta, *_ = np.linalg.lstsq(features[train], a[train], rcond=None)
            out[test] = features[test] @ beta
        return np.clip(out, 0.0, None)

    def report(name: str, weights: np.ndarray | None) -> None:
        pred = xs if weights is None else xs * weights
        pds = perturbation_discrimination(pred, ys, own_columns=own)
        scale = float((pred * ys).sum() / max((pred * pred).sum(), 1e-12))
        err = float(((scale * pred - ys) ** 2).sum() / (ys**2).sum())
        print(f"  {name:<44}{pds:>8.4f}{err:>14.4f}")

    print("\ntransfer K562 -> RPE1 with per-gene weights (genes weighted by a model that never saw them)")
    print(f"  {'':<44}{'PDS':>8}{'best-scale err':>14}   (chance PDS 0.5; err 1.0 = predicting nothing)")
    report("no weights", None)
    for name, features in feature_sets.items():
        report(f"weights from {name}", held_out_weights(features))
    report("oracle weights (true agreement, upper bound)", np.clip(a, 0.0, None))
    shuffled = [held_out_weights(design(available + [cross[p], within[p]]))
                for p in (rng.permutation(len(a)) for _ in range(20))]
    pds_null = [perturbation_discrimination(xs * w, ys, own_columns=own) for w in shuffled]
    print(f"  {'scGPT features shuffled across genes (20 draws)':<44}{np.mean(pds_null):>8.4f}   sd {np.std(pds_null):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
