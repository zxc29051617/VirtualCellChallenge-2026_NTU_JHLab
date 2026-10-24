"""Scaled (from_replicate, as on the leaderboard) member scores and their mean per subset: A current vs B trained."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
b = sys.argv[1] if len(sys.argv) > 1 else "trained_b1min"
tot = {"A": [], "B": []}; per = {"A": {}, "B": {}}
for k in range(5):
    fa, fb = f"{HERE}/mock/mock_current_s{k}.json", f"{HERE}/mock/mock_{b}_s{k}.json"
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    row = []
    for lab, f in (("A", fa), ("B", fb)):
        sc = {r["metric"]: r["from_replicate"] for r in json.load(open(f))["scaled"]}
        m = np.mean([sc[x] for x in NAMES]); tot[lab].append(m)
        for x, n in NAMES.items(): per[lab].setdefault(n, []).append(sc[x])
        row.append((lab, m, sc))
    (la, ma, sa), (lb, mb, sb) = row
    print(f"s{k}: mean A {ma:.4f} -> B {mb:.4f} ({mb - ma:+.4f}) | " + " ".join(f"{NAMES[x]} {sa[x]:+.3f}->{sb[x]:+.3f}" for x in NAMES))
if tot["A"]:
    print(f"ALL {len(tot['A'])} subsets: mean A {np.mean(tot['A']):.4f} -> B {np.mean(tot['B']):.4f} ({np.mean(tot['B']) - np.mean(tot['A']):+.4f}); B better in "
          f"{sum(b_ > a_ for a_, b_ in zip(tot['A'], tot['B']))}/{len(tot['A'])} subsets")
    print("   members: " + " ".join(f"{n} {np.mean(per['A'][n]):+.3f}->{np.mean(per['B'][n]):+.3f}" for n in per["A"]))
