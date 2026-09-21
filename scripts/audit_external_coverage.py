"""Count new-data target/context coverage from saved AUTHOR metadata, with QC flags."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcc2026.io import read_name_list
from vcc2026.local_eval import load_profiles


def run():
    out = ROOT / "artifacts/offline_20260921"
    targets = set(read_name_list(ROOT / "dataset/pert_counts.csv"))
    p = load_profiles(ROOT / "external/profiles/K562gw_perturbation_means.npz")
    covered = {g for g, n in zip(p.perturbations, p.counts) if n >= 25} & targets
    with (out / "cd4_DE_stats.suppl_table.csv").open(encoding="utf-8-sig") as f:
        de = list(csv.DictReader(f))
    with (out / "cd4_sgrna_library_metadata.suppl_table.csv").open(encoding="utf-8-sig") as f:
        guides = list(csv.DictReader(f))
    library = {r["target_gene_name"] for r in guides}
    result = {"matching": "exact gene symbols; aliases not silently merged",
              "baseline_covered": sorted(covered), "cd4_library_overlap": sorted(targets & library),
              "conditions": {}}
    output_rows = []
    for condition in sorted({r["culture_condition"] for r in de}):
        rows = {r["target_contrast_gene_name"]: r for r in de
                if r["culture_condition"] == condition}
        measured = targets & rows.keys()
        adequate = {g for g in measured if float(rows[g]["n_cells_target"] or 0) >= 100}
        # A candidate filter, not proof of absent off-target effects or transferable biology.
        credible = {g for g in adequate if rows[g]["ontarget_significant"].lower() == "true"
                    and rows[g]["distal_offtarget_flag"].lower() != "true"
                    and rows[g]["neighboring_gene_KD"].lower() != "true"
                    and rows[g]["low_target_gex"].lower() != "true"}
        result["conditions"][condition] = {
            "measured_overlap": sorted(measured), "new_over_baseline": sorted(measured - covered),
            "at_least_100_cells": sorted(adequate), "candidate_qc_overlap": sorted(credible),
            "candidate_qc_new": sorted(credible - covered),
        }
        for gene in sorted(targets):
            r = rows.get(gene, {})
            output_rows.append({"target": gene, "condition": condition,
                "in_k562gw": gene in covered, "in_cd4_library": gene in library,
                "cd4_measured": gene in measured, "candidate_qc_pass": gene in credible,
                **{k: r.get(k, "") for k in ["n_cells_target", "ontarget_effect_size",
                    "ontarget_significant", "low_target_gex", "distal_offtarget_flag",
                    "neighboring_gene_KD", "n_guides", "donor_correlation_hits_mean"]}})
    result["metadata_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.glob("cd4_*.csv")}
    result["source_tree_sha"] = {
        "cd4": json.loads((out / "cd4_repo.json").read_text(encoding="utf-8-sig"))["sha"],
        "fibro": json.loads((out / "fibro_repo.txt").read_text(encoding="utf-8-sig"))["sha"],
    }
    result["union_new_measured"] = sorted(set().union(
        *(set(x["new_over_baseline"]) for x in result["conditions"].values())))
    result["union_new_candidate_qc"] = sorted(set().union(
        *(set(x["candidate_qc_new"]) for x in result["conditions"].values())))
    for cell in ("hs27", "rpe1"):
        metadata = out / f"fibro_{cell}_metadata.json"
        if metadata.exists():
            h = json.loads(metadata.read_text())
            overlap = targets & set(h["obs"]["target_gene"])
            result[f"crispra_{cell}"] = {"target_category_overlap": sorted(overlap),
                "new_category_overlap": sorted(overlap - covered),
                "response_gene_count": len(h["var"]["gene_name"]),
                "use": "separate CRISPRa prior only; not equivalent to CRISPRi effects"}
    metadata = out / "cd4_h5_metadata.json"
    if metadata.exists():
        h = json.loads(metadata.read_text())
        axis = set(read_name_list(ROOT / "dataset/gene_names.csv"))
        result["cd4_de_response_genes"] = len(h["var"]["gene_name"])
        result["cd4_de_response_overlap"] = sorted(set(h["var"]["gene_name"]) & axis)
    (out / "external_coverage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "target_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        w.writeheader()
        w.writerows(output_rows)
    print("K562gw", len(covered), "CD4 library", len(targets & library))
    for condition, counts in result["conditions"].items():
        print(condition, {k: len(v) for k, v in counts.items()})
        print("QC new:", counts["candidate_qc_new"])


if __name__ == "__main__":
    run()
