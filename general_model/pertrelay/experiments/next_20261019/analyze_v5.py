"""V5 (shared-component removal) vs V1 (exact totals), pre-registered rule in run.log (2026-10-19 17:2x)."""
import json, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); SETS = ["H1", "KOLF200", "HCT116"]; SEEDS = [0, 1, 2]
M = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae", "de_wilcoxon_direction_fidelity_yield_raw": "fid",
     "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
runs, tgt = [], []
for st in SETS:
    for v in ["V1", "V5", "V5s"]:
        for s in SEEDS:
            d = f"{HERE}/scores/{st}_{v}_seed{s}"
            sc = pd.read_csv(f"{d}/score.csv").set_index("metric")["from_replicate"]
            agg = [f for f in os.listdir(d) if f.endswith("agg_results.csv")][0]; raw = pd.read_csv(f"{d}/{agg}").set_index("statistic").loc["mean"]
            runs.append({"set": st, "variant": v, "seed": s, "avg": float(sc["avg_score"]), **{f"score_{M[k]}": float(sc[k]) for k in M if k in sc.index},
                         "raw_mse": float(raw["expr_mse_unbiased_capped_norm"]), "raw_pds": float(raw["pds_cosine"])})
            r = pd.read_csv(f"{d}/results.csv"); r = r[r.metric == "pds_cosine"]
            for _, x in r.iterrows(): tgt.append({"set": st, "variant": v, "seed": s, "target": x.perturbation, "pds": float(x.value)})
R = pd.DataFrame(runs); T = pd.DataFrame(tgt); R.to_csv(f"{HERE}/results_v5_runs.csv", index=False)
rng = np.random.default_rng(20261011)
def boot(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return round(float(v.mean()), 4), round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)
ver = {}
for st in SETS:
    Rs = R[R.set == st]; sm = T[T.set == st].groupby(["variant", "target"])["pds"].mean().unstack(0)
    mse = Rs.pivot(index="seed", columns="variant", values="raw_mse"); avg = Rs.pivot(index="seed", columns="variant", values="avg")
    d51 = boot(sm["V5"] - sm["V1"]); d5s = boot(sm["V5"] - sm["V5s"])
    c1 = bool((mse["V5"] <= mse["V1"]).all()); c2 = d51[0] >= -0.01 and d51[1] >= -0.02; c3 = bool((avg["V5"] > avg["V5s"]).all()) and d5s[1] > 0
    ver[st] = {"c1_mse": c1, "c2_pds": bool(c2), "c3_pairing": bool(c3), "avg_V1": round(float(avg["V1"].mean()), 4), "avg_V5": round(float(avg["V5"].mean()), 4),
               "avg_not_lower": bool(avg["V5"].mean() >= avg["V1"].mean()), "pds_V5_minus_V1": d51, "pds_V5_minus_V5s": d5s}
ok = sum(e["c1_mse"] and e["c2_pds"] and e["c3_pairing"] for e in ver.values()) >= 2
ok_avg = sum(e["avg_not_lower"] for e in ver.values()) >= 2
ver["PASS"] = bool(ok and ok_avg); ver["criteria_i_iii_on_2_of_3"] = bool(ok); ver["avg_not_lower_on_2_of_3"] = bool(ok_avg)
json.dump(ver, open(f"{HERE}/verdict_v5.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(R.groupby(["set", "variant"])[["avg", "score_pds", "score_mse", "score_nmae", "score_fid", "score_reach", "score_jac", "raw_mse", "raw_pds"]].mean().round(4).to_string())
print(json.dumps(ver, indent=1))
