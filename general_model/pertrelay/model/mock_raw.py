"""Raw member values of mock variants (mean over subsets), for sets whose replicate anchors are unstable (HCT116, KOLF200:
few cells per target, many targets without significant genes): mock_raw.py <subsets> <variant> [...]
Direction: pds, fid, reach, jac higher = better; mse, nmae lower = better."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
M = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
     "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
ks = [int(x) for x in sys.argv[1].split(",")]
for v in sys.argv[2:]:
    vals = {n: [] for n in M.values()}; got = 0
    for k in ks:
        f = f"{HERE}/mock/mock_{v}_s{k}.json"
        if not os.path.exists(f): continue
        r = json.load(open(f))["raw"]; got += 1
        for m, n in M.items(): vals[n].append(r.get(m, np.nan))
    if got: print(f"{v:24s} ({got}) " + "  ".join(f"{n} {np.nanmean(x):.4f}" for n, x in vals.items()))
