"""Evaluate the option-A test against the pre-agreed pass criteria (2026-10-13).

Per held-out set (H1, KOLFbig) and generation seed (0,1,2): panel raw metrics and official-scale avg (each set's own
bundle). Per target: seed-averaged paired differences, bootstrap over targets (10,000 draws, seed 20261011).
Pass criteria (all on BOTH sets):
  1. raw mse_norm (panel) of A_plain lower than base_plain in every seed;
  2. pds not clearly lower: per-target pds diff (A_plain - base_plain) mean >= -0.01 and 95% CI lower bound >= -0.02;
  3. correct pairing beats shuffled: A_plain avg > A_shuf avg in every seed, and per-target pds diff
     (A_plain - A_shuf) has a 95% CI above 0.
"""
import json, os
import numpy as np, pandas as pd

L = "/home/zxc29051617/VCC/data/stage3/loco"; OUT = os.path.dirname(os.path.abspath(__file__))
VARS = ["base_plain", "base_shuf", "A_plain", "A_shuf"]; SEEDS = [0, 1, 2]
KEYS = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
        "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}


def boot(v, rng):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(v))


rep = {}; rows = []
for st in ["H1", "KOLFbig"]:
    per_target = {}
    for v in VARS:
        for s in SEEDS:
            d = f"{L}/{st}/score_effectfile_{st}_at_{v}_s{s}"
            if not os.path.exists(f"{d}/score.csv"): continue
            sc = pd.read_csv(f"{d}/score.csv").set_index("metric")["from_replicate"]
            agg = pd.read_csv(f"{d}/agg_results.csv").set_index("statistic").loc["mean"]
            rows.append({"set": st, "variant": v, "seed": s, "avg": float(sc["avg_score"]),
                         **{f"raw_{KEYS[k]}": float(agg[k]) for k in KEYS}})
            r = pd.read_csv(f"{d}/results.csv"); r = r[r.metric == "pds_cosine"]
            per_target[(v, s)] = r.set_index("perturbation")["value"].astype(float)
    P = pd.DataFrame(rows); P = P[P.set == st]
    if len(P) < 12: rep[st] = {"complete": False, "n_runs": len(P)}; continue
    piv = lambda col: P.pivot(index="seed", columns="variant", values=col)
    mse, avg = piv("raw_mse"), piv("avg")
    tdf = pd.DataFrame({f"{v}|{s}": per_target[(v, s)] for v in VARS for s in SEEDS})
    seedmean = {v: tdf[[f"{v}|{s}" for s in SEEDS]].mean(1) for v in VARS}
    rng = np.random.default_rng(20261011)
    pds_A_vs_base = boot(seedmean["A_plain"] - seedmean["base_plain"], rng)
    pds_A_vs_shuf = boot(seedmean["A_plain"] - seedmean["A_shuf"], rng)
    pds_base_vs_shuf = boot(seedmean["base_plain"] - seedmean["base_shuf"], rng)
    c1 = bool((mse["A_plain"] < mse["base_plain"]).all())
    c2 = bool(pds_A_vs_base[0] >= -0.01 and pds_A_vs_base[1] >= -0.02)
    c3 = bool((avg["A_plain"] > avg["A_shuf"]).all() and pds_A_vs_shuf[1] > 0)
    rep[st] = {"complete": True,
               "raw_mse_by_seed": mse.round(4).to_dict(), "avg_by_seed": avg.round(4).to_dict(),
               "pds_target_diff_A_minus_base (mean, lo, hi, n)": pds_A_vs_base,
               "pds_target_diff_A_minus_Ashuf": pds_A_vs_shuf,
               "pds_target_diff_base_minus_baseshuf": pds_base_vs_shuf,
               "criterion1_mse_improves": c1, "criterion2_pds_not_clearly_lower": c2, "criterion3_pairing_beats_shuffle": c3}
pd.DataFrame(rows).to_csv(f"{OUT}/runs.csv", index=False)
rep["PASS"] = all(rep.get(st, {}).get(f"criterion{i}_{n}", False) for st in ["H1", "KOLFbig"]
                  for i, n in [(1, "mse_improves"), (2, "pds_not_clearly_lower"), (3, "pairing_beats_shuffle")])
json.dump(rep, open(f"{OUT}/verdict.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
