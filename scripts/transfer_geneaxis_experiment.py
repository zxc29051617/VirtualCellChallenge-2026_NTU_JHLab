"""One-variable gene-axis experiment anchored to the immutable transfer_v2 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcc2026.baseline import generic_perturbation_response
from vcc2026.cell_validation import read_groups, read_rows
from vcc2026.io import (
    control_gene_profile,
    read_categorical,
    read_name_list,
    read_var_names,
    select_within_budget,
)
from vcc2026.local_eval import fold_change, load_profiles, score
from vcc2026.submission import ContextPrediction, write_perturbation_h5ad
from vcc2026.transfer import source_fold_changes

CFG = ROOT / "experiments/transfer_v3_geneaxis.json"
OUT = ROOT / "artifacts/transfer_v3_geneaxis_audit"


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(cfg):
    genes = read_name_list(ROOT / "dataset/gene_names.csv")
    perts = read_name_list(ROOT / "dataset/pert_counts.csv")
    fc, covered = source_fold_changes(ROOT / cfg["source"], perts, genes)
    fallback = generic_perturbation_response([ROOT / p for p in cfg["fallback_profiles"]], genes)
    predictions, legacy_columns = {}, {}
    for ctx in "ABC":
        path = ROOT / f"dataset/context_{ctx}.h5ad"
        assert read_var_names(path) == genes
        mean, _, _ = control_gene_profile(path)
        mean = mean * cfg["profile_scale"]
        cols = select_within_budget(mean, entries_per_cell=cfg["parent_entries_per_cell"],
                                   min_mean=1 / cfg["cell_patterns"])
        # Freeze the OLD v2 fallback assignment, even if new genes carry extra information.
        missing = ~fc[:, cols].any(1)
        shifts = fc * cfg["scale"]
        shifts[missing] = fallback
        means = (mean[None] * np.exp(shifts)).astype(np.float32)
        predictions[ctx] = ContextPrediction(ctx, np.arange(len(genes)), means)
        legacy_columns[ctx] = cols
        print("prepared", ctx, "v2 columns", len(cols), "full axis", len(genes), flush=True)
    return genes, perts, predictions, legacy_columns, int(covered.sum())


def build(cfg):
    assert digest(ROOT / cfg["protected_submission"]) == cfg["protected_sha256"]
    output = ROOT / cfg["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    genes, perts, predictions, cols, covered = prepare(cfg)
    # Before a full build, prove that the v2 reconstruction matches its actual group totals.
    parent_checks = []
    with h5py.File(ROOT / cfg["parent"], "r") as f:
        ptr = f["X/indptr"][:]
        for c, ctx in enumerate("ABC"):
            for p in (0, 149, 299):
                start = (c * len(perts) + p) * 400
                actual = read_rows(ROOT / cfg["parent"], np.arange(start, start + 400)).sum(0)
                expected = np.zeros(len(genes))
                values = predictions[ctx].means[p, cols[ctx]].astype(float)
                expected[cols[ctx]] = (np.floor(values) * 200
                    + np.rint((values - np.floor(values)) * 200)) * 2
                assert np.array_equal(actual, expected), (ctx, perts[p], np.max(abs(actual-expected)))
                parent_checks.append({"context": ctx, "target": perts[p], "exact_totals": True,
                                      "nnz": int(ptr[start + 400] - ptr[start])})
    save("build_config.json", {"config": cfg, "parent_sample_checks": parent_checks,
         "covered_targets": covered, "parent_h5ad_sha256": digest(ROOT / cfg["parent"]),
         "legacy_columns": {k: v.tolist() for k, v in cols.items()},
         "output_gene_count": len(genes)})
    print("parent reconstruction exact; writing candidate", flush=True)
    write_perturbation_h5ad(output, gene_names=genes, perturbations=perts,
        predictions=predictions, cells_per_pert=cfg["cells_per_pert"],
        cell_patterns=cfg["cell_patterns"], seed=cfg["seed"])
    print("written", output, output.stat().st_size, flush=True)


def verify(cfg):
    genes, perts, predictions, legacy, _ = prepare(cfg)
    path = ROOT / cfg["output"]
    assert read_var_names(path) == genes
    contexts, ctx_labels = read_categorical(path, "context")
    targets, target_labels = read_categorical(path, "target_gene")
    assert ctx_labels == list("ABC") and target_labels == perts
    assert np.array_equal(contexts, np.repeat(np.arange(3), 120000))
    assert np.array_equal(targets, np.tile(np.repeat(np.arange(300), 400), 3))
    results = []
    with h5py.File(path, "r") as f, h5py.File(ROOT / cfg["parent"], "r") as parent:
        assert tuple(f["X"].attrs["shape"]) == (360000, 18533)
        assert f["obs/_index/values"].shape == (360000,)
        names = f["obs/_index/values"].asstr()[:]
        assert len(set(names)) == len(names)
        # Keep dataset handles open: repeatedly opening a 360k-map VDS is expensive.
        data, indices, ptr = f["X/data"], f["X/indices"], f["X/indptr"][:]
        parent_ptr = parent["X/indptr"][:]
        for c, ctx in enumerate("ABC"):
            pd = parent[f"uns/_context_cells/{ctx}_data"]
            pi = parent[f"uns/_context_cells/{ctx}_indices"]
            parent_base = int(parent_ptr[c * 120000])
            for p, target in enumerate(perts):
                start = (c * 300 + p) * 400
                lo, hi = int(ptr[start]), int(ptr[start + 400])
                cells = sparse.csr_matrix((data[lo:hi], indices[lo:hi],
                    ptr[start:start + 401] - lo), shape=(400, len(genes))).toarray()
                assert np.isfinite(cells).all() and (cells >= 0).all()
                assert np.array_equal(cells, np.floor(cells))
                assert np.array_equal(cells[:200], cells[200:])
                values = predictions[ctx].means[p].astype(float)
                expected = (np.floor(values) * 200
                    + np.rint((values - np.floor(values)) * 200)) * 2
                assert np.array_equal(cells.sum(0), expected), (ctx, target)
                old_lo = (int(parent_ptr[start]) - parent_base) // 2
                old_hi = (int(parent_ptr[start + 400]) - parent_base) // 2
                old_totals = np.bincount(pi[old_lo:old_hi], weights=pd[old_lo:old_hi],
                                        minlength=len(genes)) * 2
                assert np.array_equal(cells.sum(0)[legacy[ctx]], old_totals[legacy[ctx]])
                assert cells.sum(1).max() <= 1000000
                results.append({"context": ctx, "target": target, "cells": 400,
                    "nonzero_genes": int((cells.sum(0) > 0).sum()),
                    "max_library": float(cells.sum(1).max()), "totals_exact": True,
                    "v2_retained_gene_totals_exact": True})
                if p % 50 == 0:
                    print("verified", ctx, p, "/300", flush=True)
        nnz = int(f["X/indptr"][-1])
    save("validation.json", {"shape": [360000, 18533], "groups": results,
        "gene_order_exact": True, "metadata_exact": True, "nnz": nnz,
        "h5ad_bytes": path.stat().st_size, "h5ad_sha256": digest(path),
        "protected_v2_sha256": digest(ROOT / cfg["protected_submission"])})
    assert digest(ROOT / cfg["protected_submission"]) == cfg["protected_sha256"]


def evaluate(cfg):
    records = []
    prior = json.loads((ROOT / "artifacts/offline_20260921/roundtrip.json").read_text())
    for source, target in [("K562", "RPE1"), ("RPE1", "K562")]:
        s, t = [load_profiles(ROOT / f"external/profiles/{c}_perturbation_means.npz")
                for c in (source, target)]
        genes = sorted(set(s.genes) & set(t.genes))
        sc, tc = [[p.genes.index(g) for g in genes] for p in (s, t)]
        # Exact previously evaluated split, without resampling or parameter fitting.
        perts = next(r["targets"] for r in prior if r["source"] == source)
        ctrl = t.means[t.index_of("control"), tc].astype(float)
        truth = t.means[np.ix_([t.index_of(p) for p in perts], tc)]
        fc = fold_change(s.means[np.ix_([s.index_of(p) for p in perts], sc)],
                         s.means[s.index_of("control"), sc])
        means = ctrl[None] * np.exp(fc * cfg["scale"]) * cfg["profile_scale"]
        legacy = select_within_budget(ctrl * cfg["profile_scale"],
                    entries_per_cell=cfg["parent_entries_per_cell"], min_mean=1/200)
        for method, cols in [("v2_gene_strategy", legacy),
                ("full_common_axis", np.arange(len(genes))),
                ("top3000_matched_parameters", np.argsort(-ctrl)[:3000])]:
            path = OUT / f"{source}_{target}_{method}.h5ad"
            write_perturbation_h5ad(path, gene_names=genes, perturbations=perts,
                predictions={target: ContextPrediction(target, cols, means[:, cols])},
                cells_per_pert=400, cell_patterns=200, seed=0)
            actual = read_groups(path, genes)
            values = np.stack([actual[p].mean(0) for p in perts])
            metrics = score(values, truth, ctrl, perturbations=perts, genes=genes)
            row = {"source": source, "target": target, "method": method,
                "targets": perts, "genes": len(genes), "selected": len(cols),
                "profile_scale": .15, "effect_scale": .3, "patterns": 200, "cells": 400,
                "seed": 0, "scores": metrics, "path": str(path)}
            records.append(row)
            save("evaluation.json", records)
            print(source, target, method, metrics, flush=True)


def archive(cfg):
    """Verify the actual packed payload, not just the pre-prep prediction."""
    import shutil
    import tarfile

    import zstandard

    packed = ROOT / cfg["submission"]
    extracted = OUT / "prepared.h5ad"
    with tarfile.open(packed, "r:") as tar:
        assert tar.getnames() == ["pred.h5ad.zst"]
        with (tar.extractfile("pred.h5ad.zst") as src, open(extracted, "wb") as dst,
              zstandard.ZstdDecompressor().stream_reader(src) as reader):
            shutil.copyfileobj(reader, dst, length=8 * 1024**2)
    original = ROOT / cfg["output"]
    genes = read_name_list(ROOT / "dataset/gene_names.csv")
    assert read_var_names(extracted) == genes
    for column in ("context", "target_gene"):
        codes, labels = read_categorical(original, column)
        other_codes, other_labels = read_categorical(extracted, column)
        # AnnData can reorder categorical dictionaries during prep; row labels must match.
        assert np.array_equal(np.asarray(labels)[codes], np.asarray(other_labels)[other_codes])
    hashes = {}
    with h5py.File(original, "r") as a, h5py.File(extracted, "r") as b:
        assert tuple(a["X"].attrs["shape"]) == tuple(b["X"].attrs["shape"])
        assert not b["X/data"].is_virtual and not b["X/indices"].is_virtual
        for field, dtype in [("indptr", "int64"), ("indices", "int32"), ("data", "float32")]:
            first, second = a[f"X/{field}"], b[f"X/{field}"]
            assert first.shape == second.shape
            h = hashlib.sha256()
            for start in range(0, first.size, 8_000_000):
                av = first[start:start+8_000_000].astype(dtype)
                bv = second[start:start+8_000_000].astype(dtype)
                assert np.array_equal(av, bv), (field, start)
                h.update(av.tobytes())
            hashes[field] = h.hexdigest()
            print("packed payload exact", field, first.size, flush=True)
    protected = digest(ROOT / cfg["protected_submission"])
    assert protected == cfg["protected_sha256"]
    parent_digest = digest(ROOT / cfg["parent"])
    assert parent_digest == json.loads((OUT / "build_config.json").read_text())["parent_h5ad_sha256"]
    save("archive_validation.json", {"submission": str(packed),
        "vcc_bytes": packed.stat().st_size, "vcc_sha256": digest(packed),
        "prepared_h5ad_bytes": extracted.stat().st_size, "csr_exact": True,
        "metadata_exact": True, "gene_order_exact": True, "self_contained": True,
        "canonical_csr_sha256": hashes, "protected_v2_sha256": protected,
        "protected_parent_h5ad_sha256": parent_digest,
        "config_sha256": digest(CFG), "driver_sha256": digest(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["build", "verify", "evaluate", "archive"])
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    {"build": build, "verify": verify, "evaluate": evaluate, "archive": archive}[args.stage](cfg)
