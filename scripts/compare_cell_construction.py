"""Distribution-only development experiment: exact same quantized means as v3.

Uses the saved 16-target bidirectional single-cell splits. No new data or neural models.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcc2026.cell_validation import differential_expression, read_groups, write_cells
from vcc2026.local_eval import fold_change, score
from vcc2026.submission import cells_from_mean

OUT = ROOT / "artifacts/cell_construction_followup"


def fixed_total_cells(mean, controls, method, seed):
    """200 patterns, repeated twice; all methods preserve EXACT floor/ceil totals."""
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean, dtype=float)
    totals = (np.floor(mean) * 200 + np.rint((mean - np.floor(mean)) * 200)).astype(int)
    if method == "floorceil":
        patterns = cells_from_mean(mean, 200, rng)
    else:
        m = controls.mean(0, dtype=float)
        v = controls.var(0, ddof=1, dtype=float)
        theta = np.divide(m**2, v-m, out=np.full_like(m, 1e6), where=v > m+1e-10)
        theta = np.clip(theta, .001, 1e6)
        # Source of shared cell state for the empirical method; identical rows across genes.
        template = controls[rng.choice(len(controls), 200, replace=True)]
        patterns = np.zeros((200, len(mean)), dtype=np.float32)
        for j, total in enumerate(totals):
            if not total:
                continue
            if method == "conditioned_nb":
                # Raw-control theta is invariant under binomial count thinning.
                weights = rng.gamma(theta[j], 1, 200)
            elif method == "empirical_weights":
                weights = template[:, j].astype(float)
            else:
                raise ValueError(method)
            if weights.sum() == 0:
                weights = np.ones(200)
            patterns[:, j] = rng.multinomial(total, weights / weights.sum())
    assert np.array_equal(patterns.sum(0), totals)
    return np.tile(patterns, (2, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = {}
    for name in ("transfer_v2.vcc", "transfer_v3_geneaxis.vcc"):
        with open(ROOT / "artifacts" / name, "rb") as f:
            protected[name] = hashlib.file_digest(f, "sha256").hexdigest()
    records = []
    for source, target in [("K562", "RPE1"), ("RPE1", "K562")]:
        path = ROOT / f"artifacts/offline_20260921/pilot_cells_{source}_{target}_16.npz"
        data = np.load(path, allow_pickle=False)
        genes, perts = data["genes"].tolist(), data["perts"].tolist()
        controls = data["target_controls"][:400]
        reference = data["target_controls"][400:672]
        real_null = data["target_controls"][672:]
        truth = data["target_cells"]
        source_means = data["source_cells"][:, :96].mean(1)
        source_control = data["source_controls"][:400].mean(0)
        effects = fold_change(source_means, source_control)
        means = controls.mean(0)[None] * np.exp(.3 * effects) * .15
        truth_de = [differential_expression(x, reference,
            [genes.index(p)] if p in genes else []) for p, x in zip(perts, truth)]
        null_truth = differential_expression(real_null, reference)
        for seed in (0, 1, 2):
            baseline_pds = None
            for method in ("floorceil", "conditioned_nb", "empirical_weights"):
                groups = {p: fixed_total_cells(means[i], controls, method, seed*1000+i)
                          for i, p in enumerate(perts)}
                groups["null_diagnostic"] = fixed_total_cells(
                    controls.mean(0) * .15, controls, method, seed*1000+999)
                output = OUT / f"{source}_{target}_{method}_{seed}.h5ad"
                write_cells(output, groups, genes, target)
                actual = read_groups(output, genes)
                readback_means = np.stack([actual[p].mean(0) for p in perts])
                scores = score(readback_means, truth.mean(1), reference.mean(0),
                               perturbations=perts, genes=genes)
                if baseline_pds is None:
                    baseline_pds = scores["pds"]
                assert scores["pds"] == baseline_pds
                details = []
                for i, p in enumerate(perts):
                    d = differential_expression(actual[p], reference,
                        [genes.index(p)] if p in genes else [])
                    a, b = d["called"], truth_de[i]["called"]
                    k = int((a & b).sum())
                    details.append({"target": p, "pred_de": int(a.sum()),
                        "true_de": int(b.sum()), "intersection": k,
                        "precision": k/max(int(a.sum()), 1), "recall": k/max(int(b.sum()), 1),
                        "jaccard": k/max(int((a | b).sum()), 1)})
                null = differential_expression(actual["null_diagnostic"], reference)
                result = {"source": source, "target": target, "method": method, "seed": seed,
                    "scores": scores, "details": details,
                    "mean_jaccard": float(np.mean([x["jaccard"] for x in details])),
                    "mean_precision": float(np.mean([x["precision"] for x in details])),
                    "mean_recall": float(np.mean([x["recall"] for x in details])),
                    "null_de": int(null["called"].sum()),
                    "real_null_de": int(null_truth["called"].sum()),
                    "mean_totals_exact": True, "h5ad": str(output)}
                records.append(result)
                (OUT / "results.json").write_text(json.dumps(records, indent=2))
                print(source, target, seed, method, "PDS", scores["pds"],
                      "DE-jac", result["mean_jaccard"], "null DE", result["null_de"], flush=True)
    for name, expected in protected.items():
        with open(ROOT / "artifacts" / name, "rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    (OUT / "manifest.json").write_text(json.dumps({"protected": protected,
        "runs": len(records), "profile_scale": .15, "effect_scale": .3,
        "patterns": 200, "predicted_cells": 400, "truth_cells": 128,
        "fit_controls": 400, "reference_controls": 272, "real_null_controls": 128,
        "scope": "development diagnostic; not official metrics; no v4 selected"}, indent=2))


if __name__ == "__main__":
    main()
