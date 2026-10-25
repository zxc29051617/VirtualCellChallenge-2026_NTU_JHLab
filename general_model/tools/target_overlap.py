"""Overlap of perturbation gene lists with the 300 D/E/F targets (2026-10-25).

    python3 -I tools/target_overlap.py <dir with one-symbol-per-line .txt lists> [targets.csv]

targets.csv defaults to data/vcc2026_final/pert_counts.csv (the final-phase download, README section 11.5).

Exact HGNC symbol match; FAM200B (a target) is matched under its current symbol ZBED11 as well.
"""
import csv, glob, os, sys

d = sys.argv[1]
tf = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vcc2026_final", "pert_counts.csv")
T = {r["target_gene"].strip() for r in csv.DictReader(open(tf))} - {"non-targeting", ""}
alias = {"ZBED11": "FAM200B"}
res = []
for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
    L = {alias.get(g, g) for g in (l.strip() for l in open(f, errors="replace")) if g}
    res.append((len(T & L), os.path.basename(f), len(L)))
for n, f, k in sorted(res, reverse=True):
    print(f"{n:4d}/{len(T)}  {f:62s} ({k} genes)")
