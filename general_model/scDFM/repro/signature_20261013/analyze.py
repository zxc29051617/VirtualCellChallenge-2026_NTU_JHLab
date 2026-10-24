"""Statistics for the source-weight experiment (run after evaluate_runner.py; works on whatever is finished).

    python analyze.py                (evalenv or scDFM venv; pandas/numpy/pyarrow)

Outputs: results_per_run.csv (panel raw metrics + official-scale scores per set/group/seed), results_per_target.csv,
paired_seedmean.csv, bootstrap.csv (target-level paired differences: seeds averaged first, targets are the sampling
unit, 10,000 draws, rng seed 20261011), inner_cv.csv, verdict.json. The official MSE member (expr_mse_unbiased_capped_norm)
exists only at panel level: it is reported per run and never averaged from per-target values; per-target tables only carry
error diagnostics (expr_mse_unbiased_capped). `fid_scaled_heuristic_b05_NOT_OFFICIAL` is a heuristic column only.

Pre-set decision rule ("SIGnature adds value over expression features"), required in >= 2 of the 3 outer sets:
  (i)  panel raw mse_norm of C below both A and B in all 3 generation seeds;
  (ii) per-target pds difference C - B: mean >= -0.01 and 95% CI lower bound >= -0.02 (PDS preserved);
  (iii) C beats D: panel raw mse lower in all 3 seeds AND per-target pds difference C - D has mean > 0.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"; sys.path.insert(0, HERE)
SETS = ["H1", "HCT116", "KOLF200"]; GROUPS = ["A", "B", "C", "D", "A_shuf"]; SEEDS = [0, 1, 2]; LOCO = {"H1": "H1", "HCT116": "HCT116", "KOLF200": "KOLF200"}
M = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae", "de_wilcoxon_direction_fidelity_yield_raw": "fid",
     "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
PT = {"pds_cosine": "pds", "de_wilcoxon_lfc_nmae": "nmae", "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach",
      "de_wilcoxon_sig_jaccard": "jac", "expr_mse_unbiased_capped": "diag_mse_capped"}
LOW = {"nmae", "diag_mse_capped"}
COMPARE = [("C", "A"), ("C", "B"), ("C", "D"), ("B", "A"), ("D", "B"), ("A", "A_shuf"), ("C", "A_shuf")]


def boot(v, rng):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(10000)])
    return v.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), len(v)


runs, tgt = [], []
for st in SETS:
    D = f"{ROOT}/data/stage3/loco/{LOCO[st]}/bundle"
    base = {M[k]: float(v) for k, v in pd.read_csv(f"{D}/baseline_agg.csv").set_index("statistic").loc["mean"].items() if k in M}
    rep = {M[k]: float(v) for k, v in pd.read_parquet(f"{D}/anchor_agg.parquet").set_index("metric")["replicate"].items() if k in M}
    for g in GROUPS:
        for s in SEEDS:
            d = f"{HERE}/scores/{st}_{g}_seed{s}"
            if not os.path.exists(f"{d}/score.csv"): continue
            sc = pd.read_csv(f"{d}/score.csv").set_index("metric")["from_replicate"]
            raw = {M[k]: float(v) for k, v in pd.read_csv(f"{d}/agg_results.csv").set_index("statistic").loc["mean"].items() if k in M}
            b2 = max(base["fid"], 0.5)
            runs.append({"set": st, "group": g, "seed": s, "avg_official_scale": float(sc["avg_score"]), **{f"score_{M[k]}": float(sc[k]) for k in M if k in sc.index},
                         **{f"panel_raw_{k}": v for k, v in raw.items()}, "fid_scaled_heuristic_b05_NOT_OFFICIAL": (raw["fid"] - b2) / (rep["fid"] - b2)})
            r = pd.read_csv(f"{d}/results.csv"); r = r[r.metric.isin(PT)]
            for _, x in r.iterrows():
                tgt.append({"set": st, "group": g, "seed": s, "target": x.perturbation, "metric": PT[x.metric], "raw": float(x.value) if pd.notna(x.value) else np.nan})
R = pd.DataFrame(runs); T = pd.DataFrame(tgt)
R.to_csv(f"{HERE}/results_per_run.csv", index=False); T.to_csv(f"{HERE}/results_per_target.csv", index=False)
verdict = {"n_runs_scored": int(len(R)), "sets": {}}
rng = np.random.default_rng(20261011); rows = []; sm_rows = []
for st in SETS:
    Rs = R[R.set == st] if len(R) else R; Ts = T[T.set == st] if len(T) else T
    if len(Rs) == 0: continue
    full = {g for g in GROUPS if len(Rs[Rs.group == g]) == len(SEEDS)}
    sm = Ts.groupby(["group", "target", "metric"])["raw"].mean().reset_index()
    sv = {}
    for a, b in COMPARE:
        if a not in full or b not in full: continue
        pa = sm[sm.group == a].set_index(["target", "metric"])["raw"]; pb = sm[sm.group == b].set_index(["target", "metric"])["raw"]
        diff = (pa - pb).dropna().reset_index()
        for metric, grp in diff.groupby("metric"):
            m, lo, hi, n = boot(grp["raw"].values, rng)
            better = (grp["raw"] < 0) if metric in LOW else (grp["raw"] > 0)
            rows.append({"set": st, "comparison": f"{a}-{b}", "metric": metric, "mean_diff": m, "ci95_lo": lo, "ci95_hi": hi, "n_targets": n, "targets_first_better": int(better.sum()), "lower_is_better": metric in LOW})
            sv[(a, b, metric)] = (m, lo, hi)
    panel = Rs.pivot(index="seed", columns="group", values="panel_raw_mse")
    entry = {"complete_groups": sorted(full)}
    if {"A", "B", "C", "D"} <= full:
        c1 = bool(((panel["C"] < panel["A"]) & (panel["C"] < panel["B"])).all())
        pd_cb = sv.get(("C", "B", "pds")); c2 = bool(pd_cb and pd_cb[0] >= -0.01 and pd_cb[1] >= -0.02)
        pd_cd = sv.get(("C", "D", "pds")); c3 = bool(pd_cd and (panel["C"] < panel["D"]).all() and pd_cd[0] > 0)
        entry.update({"panel_raw_mse_by_seed": panel.round(4).to_dict(), "criterion_i_mse_below_A_and_B_all_seeds": c1,
                      "criterion_ii_pds_preserved_vs_B": c2, "criterion_iii_C_beats_D": c3, "pds_C_minus_B": pd_cb, "pds_C_minus_D": pd_cd,
                      "avg_by_group_mean_over_seeds": Rs.groupby("group")["avg_official_scale"].mean().round(4).to_dict()})
    verdict["sets"][st] = entry
pd.DataFrame(rows).to_csv(f"{HERE}/bootstrap.csv", index=False)
done = [e for e in verdict["sets"].values() if "criterion_i_mse_below_A_and_B_all_seeds" in e]
if len(done) == len(SETS):
    ok = [e["criterion_i_mse_below_A_and_B_all_seeds"] and e["criterion_ii_pds_preserved_vs_B"] and e["criterion_iii_C_beats_D"] for e in done]
    verdict["all_criteria_met_in_sets"] = ok; verdict["SIGnature_adds_value"] = bool(sum(ok) >= 2)
# inner CV table from the weight-learning step
ic = []
for st in SETS:
    f = f"{HERE}/weights_{st}.json"
    if os.path.exists(f):
        W = json.load(open(f))
        for g in ["B", "C", "D"]:
            for k, v in W[g]["inner_cv"].items(): ic.append({"outer": st, "group": g, "kappa": k, **v, "chosen": float(k) == W[g]["kappa"]})
pd.DataFrame(ic).to_csv(f"{HERE}/inner_cv.csv", index=False)
json.dump(verdict, open(f"{HERE}/verdict.json", "w"), indent=1, default=float)
if len(R):
    pd.set_option("display.width", 250)
    print(R.groupby(["set", "group"])[["avg_official_scale", "score_pds", "score_mse", "panel_raw_mse", "panel_raw_pds", "score_nmae", "score_fid", "score_reach", "score_jac"]].mean().round(4).to_string())
print(json.dumps(verdict, indent=1, default=float)[:3000])
