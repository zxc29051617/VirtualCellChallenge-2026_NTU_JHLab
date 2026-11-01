"""Scaled (leaderboard) member scores of mock-exam variants side by side: mock_cmp.py <subsets e.g. 0,1> <variant> [<variant> ...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
ks = [int(x) for x in sys.argv[1].split(",")]
for v in sys.argv[2:]:
    per = {n: [] for n in NAMES.values()}; tot = []
    for k in ks:
        f = f"{HERE}/mock/mock_{v}_s{k}.json"
        if not os.path.exists(f): continue
        sc = {r["metric"]: r["from_replicate"] for r in json.load(open(f))["scaled"]}
        for x, n in NAMES.items(): per[n].append(sc[x])
        tot.append(np.mean([sc[x] for x in NAMES]))
    if tot:
        print(f"{v:22s} overall {np.mean(tot):.4f} ({len(tot)} subsets) | " + " ".join(f"{n} {np.mean(x):+.3f}" for n, x in per.items()))
