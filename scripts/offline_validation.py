"""Reproducible output roundtrip and independent-control null experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcc2026.cell_validation import (
    construct_cells,
    differential_expression,
    read_groups,
    read_rows,
    write_cells,
)
from vcc2026.io import read_var_names
from vcc2026.local_eval import fold_change, load_profiles, score
from vcc2026.submission import ContextPrediction, write_perturbation_h5ad


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def roundtrip(out, limit):
    records = []
    for source, target in [("K562", "RPE1"), ("RPE1", "K562")]:
        s, t = [load_profiles(ROOT / f"external/profiles/{c}_perturbation_means.npz")
                for c in (source, target)]
        genes = sorted(set(s.genes) & set(t.genes))
        sc, tc = [[p.genes.index(g) for g in genes] for p in (s, t)]
        perts = sorted(set(s.perturbations) & set(t.perturbations) - {"control"})
        perts = [p for p in perts if min(s.counts[s.index_of(p)], t.counts[t.index_of(p)]) >= 50]
        rng = np.random.default_rng(20260921)
        perts = sorted(rng.choice(perts, min(limit, len(perts)), replace=False).tolist())
        ctrl = t.means[t.index_of("control"), tc].astype(float)
        truth = t.means[np.ix_([t.index_of(p) for p in perts], tc)]
        fc = fold_change(s.means[np.ix_([s.index_of(p) for p in perts], sc)],
                         s.means[s.index_of("control"), sc])
        means = ctrl[None] * np.exp(fc * 0.3)
        for name, cols, depth, patterns in [
            ("all_continuous", np.arange(len(genes)), 1., 400),
            ("all_integer400", np.arange(len(genes)), 1., 400),
            ("all_depth015_integer400", np.arange(len(genes)), .15, 400),
            ("all_depth015_integer200", np.arange(len(genes)), .15, 200),
            ("top3000_integer400", np.argsort(-ctrl)[:3000], 1., 400),
            ("de_gate_integer400", np.flatnonzero(ctrl / ctrl.sum() * 1e6 > 5), 1., 400),
        ]:
            continuous = means * depth
            if name == "all_continuous":
                actual = continuous
                filename = None
            else:
                filename = out / f"roundtrip_{source}_{target}_{name}.h5ad"
                write_perturbation_h5ad(filename, gene_names=genes, perturbations=perts,
                    predictions={target: ContextPrediction(target, cols, continuous[:, cols])},
                    cell_patterns=patterns)
                groups = read_groups(filename, genes)
                actual = np.stack([groups[p].mean(0) for p in perts])
            result = {"source": source, "target": target, "method": name, "targets": perts,
                "genes": len(genes), "selected_genes": len(cols), "patterns": patterns, "depth": depth,
                "output": str(filename) if filename else None,
                "mean_abs_roundtrip_error": float(np.abs(actual - continuous).mean()),
                "scores": score(actual, truth, ctrl, perturbations=perts, genes=genes)}
            records.append(result)
            print(source, target, name, result["scores"], flush=True)
            save(out / "roundtrip.json", records)


def null_experiment(out, seeds):
    records = []
    for context in "ABC":
        path = ROOT / f"dataset/context_{context}.h5ad"
        genes = read_var_names(path)
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            rows = rng.permutation(18400)[:3200]
            all_cells = read_rows(path, rows)
            fit, ref, real = all_cells[:2000], all_cells[2000:2800], all_cells[2800:]
            mean, var = fit.mean(0, dtype=float), fit.var(0, ddof=1, dtype=float)
            for method, depth in [("real_null", 1), ("resample", 1),
                                  ("floorceil", 1), ("floorceil", .15),
                                  ("nb", 1), ("conditioned_nb", 1)]:
                if method == "real_null":
                    cells = real
                elif method == "resample":
                    cells = fit[rng.choice(len(fit), 400, replace=True)]
                else:
                    cells = construct_cells(mean * depth, var * depth**2, 400,
                                            np.random.default_rng(seed), method)
                filename = out / f"null_{context}_{seed}_{method}_{depth}.h5ad"
                write_cells(filename, {"null": cells}, genes, context)
                actual = read_groups(filename, genes)["null"]
                de = differential_expression(actual, ref)
                tested, called = int(de["tested"].sum()), int(de["called"].sum())
                result = {"context": context, "seed": seed, "method": method, "depth": depth,
                    "tested": tested, "called": called, "rejection_fraction": called / max(tested, 1),
                    "zero_fraction": float((actual == 0).mean()),
                    "mean_abs_error": float(np.abs(actual.mean(0) - mean * depth).mean()),
                    "fit_rows": rows[:2000].tolist(), "reference_rows": rows[2000:2800].tolist(),
                    "real_null_rows": rows[2800:].tolist(), "output": str(filename)}
                records.append(result)
                print("NULL", context, seed, method, depth, called, "/", tested, flush=True)
                save(out / "null.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["roundtrip", "null"])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/offline_20260921")
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "roundtrip":
        roundtrip(args.out, args.targets)
    else:
        null_experiment(args.out, args.seeds)
