"""Re-score local check-set results with an A/B/C-like anchor for direction fidelity.

On the official val contexts the context-mean baseline already sits at the chance level of
`de_wilcoxon_direction_fidelity_yield_raw` (0.513, derived from our two 10-11 submissions): it calls many DE genes
and half of their signs agree by chance. On our local check sets the baseline calls almost nothing (raw 0.004-0.07),
so ANY submission with many DE calls collects a large scaled fidelity there, which A/B/C do not pay. This replaces
the local fidelity baseline by max(local, 0.5) and reports both averages.

    python adjusted_score.py data/stage3/loco/KOLFbig            # every score_* directory in that set
"""
import csv, glob, os, sys
import pandas as pd

M = {"pds_cosine": "high", "expr_mse_unbiased_capped_norm": "low", "de_wilcoxon_lfc_nmae": "low",
     "de_wilcoxon_direction_fidelity_yield_raw": "high", "de_wilcoxon_direction_reach_raw": "high",
     "de_wilcoxon_sig_jaccard": "high"}
FLOOR0 = {"expr_mse_unbiased_capped_norm"}               # the official mse member is clipped at 0
SHORT = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach",
         "de_wilcoxon_sig_jaccard": "jac"}


def means(path):
    return {k: float(v) for k, v in {x["statistic"]: x for x in csv.DictReader(open(path))}["mean"].items()
            if k in M and v not in ("", "NaN")}


def main(d):
    base = means(f"{d}/bundle/baseline_agg.csv")
    rep = pd.read_parquet(f"{d}/bundle/anchor_agg.parquet").set_index("metric")["replicate"].to_dict()
    rows = []
    for sd in sorted(glob.glob(f"{d}/score_*")):
        f = f"{sd}/agg_results.csv"
        if not os.path.exists(f): continue
        u = means(f); s, s_adj = {}, {}
        for m in M:
            b, a = base.get(m), rep.get(m)
            if b is None or a is None or m not in u: continue
            val = (u[m] - b) / (a - b)
            if m in FLOOR0: val = max(val, 0.0)
            s[m] = val
            if m == "de_wilcoxon_direction_fidelity_yield_raw":
                b2 = max(b, 0.5); s_adj[m] = (u[m] - b2) / (a - b2)
            else:
                s_adj[m] = val
        rows.append((os.path.basename(sd).replace("score_", ""), sum(s.values()) / len(s), sum(s_adj.values()) / len(s_adj), s_adj))
    print(f"{os.path.basename(d)}  (fid baseline local {base['de_wilcoxon_direction_fidelity_yield_raw']:.3f} -> 0.5; replicate {rep['de_wilcoxon_direction_fidelity_yield_raw']:.3f})")
    for name, avg, adj, s in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:45s} avg {avg:7.3f}  adjusted {adj:7.3f}   " + " ".join(f"{SHORT[m]} {v:6.3f}" for m, v in s.items()))


if __name__ == "__main__":
    for d in sys.argv[1:]: main(d.rstrip("/"))
