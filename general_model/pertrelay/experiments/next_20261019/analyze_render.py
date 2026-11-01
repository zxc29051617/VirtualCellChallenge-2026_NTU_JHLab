"""Statistics for the rendering experiment (render_eval.py). Works on whatever runs are finished.

V0 / V0s are the signature_20261013 group A / A_shuf runs (same model, same control draws, current renderer).
Per set: panel raw metrics and official-scale scores per seed (results_render_runs.csv); per-target paired differences
on pds and the per-target error diagnostic (seeds averaged first, targets resampled, 10,000 draws, rng 20261011;
bootstrap_render.csv). The official MSE member is panel-only and is never rebuilt from per-target values.
Pre-set pass rule (render_eval.py docstring), needed in >= 2 of 3 sets: (i) raw panel mse_norm V2 < V0 in all seeds;
(ii) per-target pds V2 - V0 mean >= -0.01 and CI low >= -0.02; (iii) V2 avg > V2s avg in all seeds and pds V2 - V2s
CI low > 0.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"; SIG = f"{ROOT}/pertrelay/experiments/signature_20261013"
SETS = ["H1", "KOLF200", "HCT116"]; SEEDS = [0, 1, 2]
SRC = {"V0": (SIG, "A"), "V0s": (SIG, "A_shuf"), "V1": (HERE, "V1"), "V2": (HERE, "V2"), "V2s": (HERE, "V2s")}
M = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae", "de_wilcoxon_direction_fidelity_yield_raw": "fid",
     "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
PT = {"pds_cosine": "pds", "expr_mse_unbiased_capped": "diag_mse_capped"}


def boot(v, rng):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(v))


runs, tgt = [], []
for st in SETS:
    for v, (base, g) in SRC.items():
        for s in SEEDS:
            d = f"{base}/scores/{st}_{g}_seed{s}"
            if not os.path.exists(f"{d}/score.csv"): continue
            sc = pd.read_csv(f"{d}/score.csv").set_index("metric")["from_replicate"]
            agg = [f for f in os.listdir(d) if f.endswith("agg_results.csv")][0]
            raw = pd.read_csv(f"{d}/{agg}").set_index("statistic").loc["mean"]
            runs.append({"set": st, "variant": v, "seed": s, "avg": float(sc["avg_score"]), **{f"score_{M[k]}": float(sc[k]) for k in M if k in sc.index},
                         **{f"raw_{M[k]}": float(raw[k]) for k in M}})
            r = pd.read_csv(f"{d}/results.csv"); r = r[r.metric.isin(PT)]
            for _, x in r.iterrows():
                tgt.append({"set": st, "variant": v, "seed": s, "target": x.perturbation, "metric": PT[x.metric], "raw": float(x.value) if pd.notna(x.value) else np.nan})
R = pd.DataFrame(runs); T = pd.DataFrame(tgt); R.to_csv(f"{HERE}/results_render_runs.csv", index=False)
rng = np.random.default_rng(20261011); rows = []; verdict = {"sets": {}}
for st in SETS:
    Rs = R[R.set == st]; Ts = T[T.set == st]
    full = {v for v in SRC if len(Rs[Rs.variant == v]) == len(SEEDS)}
    sm = Ts.groupby(["variant", "target", "metric"])["raw"].mean().reset_index(); sv = {}
    for a, b in [("V1", "V0"), ("V2", "V0"), ("V2", "V1"), ("V2", "V2s"), ("V0", "V0s")]:
        if a not in full or b not in full: continue
        pa = sm[sm.variant == a].set_index(["target", "metric"])["raw"]; pb = sm[sm.variant == b].set_index(["target", "metric"])["raw"]
        diff = (pa - pb).dropna().reset_index()
        for metric, grp in diff.groupby("metric"):
            m, lo, hi, n = boot(grp["raw"].values, rng)
            rows.append({"set": st, "comparison": f"{a}-{b}", "metric": metric, "mean_diff": m, "ci95_lo": lo, "ci95_hi": hi, "n_targets": n}); sv[(a, b, metric)] = (m, lo, hi)
    e = {"complete": sorted(full)}
    if {"V0", "V2", "V2s"} <= full:
        mse = Rs.pivot(index="seed", columns="variant", values="raw_mse"); avg = Rs.pivot(index="seed", columns="variant", values="avg")
        p20 = sv[("V2", "V0", "pds")]; p2s = sv[("V2", "V2s", "pds")]
        e.update({"raw_mse_by_seed": mse.round(4).to_dict(), "avg_by_seed": avg.round(4).to_dict(),
                  "c1_mse_V2_below_V0_all_seeds": bool((mse["V2"] < mse["V0"]).all()),
                  "c2_pds_not_clearly_lower": bool(p20[0] >= -0.01 and p20[1] >= -0.02),
                  "c3_pairing_beats_shuffle": bool((avg["V2"] > avg["V2s"]).all() and p2s[1] > 0),
                  "pds_V2_minus_V0": p20, "pds_V2_minus_V2s": p2s})
        e["all"] = e["c1_mse_V2_below_V0_all_seeds"] and e["c2_pds_not_clearly_lower"] and e["c3_pairing_beats_shuffle"]
    verdict["sets"][st] = e
pd.DataFrame(rows).to_csv(f"{HERE}/bootstrap_render.csv", index=False)
done = [e for e in verdict["sets"].values() if "all" in e]
if len(done) == len(SETS): verdict["PASS"] = bool(sum(e["all"] for e in done) >= 2)
json.dump(verdict, open(f"{HERE}/verdict_render.json", "w"), indent=1)
pd.set_option("display.width", 250)
if len(R):
    print(R.groupby(["set", "variant"])[["avg", "score_pds", "score_mse", "score_nmae", "score_fid", "score_reach", "score_jac", "raw_mse", "raw_pds"]].mean().round(4).to_string())
print(json.dumps({k: v for k, v in verdict.items()}, indent=1, default=str)[:2500])
