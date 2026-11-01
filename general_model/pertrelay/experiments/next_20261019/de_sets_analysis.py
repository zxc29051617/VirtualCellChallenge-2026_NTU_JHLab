"""How good are our predicted significant-gene sets, and what would a controlled call set achieve? (hold-outs)

    python de_sets_analysis.py [set ...]        (pertrelay venv; runs cell-eval2 once per set with --write-degenes)

For each hold-out: regenerate the V6 prediction (seed 0, same draws as the scored runs), score it once with the DE
tables written, then compare per target:
  real set R_p   genes with p_adj < 0.05 in the real data (scorer's gate, own gene excluded)
  called set     genes with p_adj < 0.05 in our predicted cells
  ranked top-K   the K = |R_p| genes with the largest |predicted log2FC| among gated genes (what a controlled call set
                 of the right size would contain); h = recall of R_p, direction accuracy on the overlap.
Uses hold-out truth: this is an analysis of headroom, not a tuned result.
"""
import glob, json, os, subprocess, sys
import numpy as np, pandas as pd, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_eval as E

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/de_sets"; os.makedirs(OUT, exist_ok=True)
res = {}
for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
    D, R, C, lib, rate, ref_mean = E.load_set(name); comp = E.components(name); targets, oc, full, shuf = comp
    od = f"{OUT}/{name}"
    if not glob.glob(f"{od}/*de*real*") and not glob.glob(f"{od}/*real*de*"):
        P = E.generate(name, 0, "V6", R, C, lib, rate, ref_mean, comp); pf = f"{OUT}/{name}_V6.h5ad"; P.write_h5ad(pf)
        subprocess.run([E.CE2, "run", "-ap", pf, "-ar", f"{D}/real.h5ad", "-o", od, *E.COMMON, "--write-degenes"], check=True,
                       env=E.ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(pf)
    files = sorted(os.listdir(od)); de = [f for f in files if "de" in f.lower() and (f.endswith(".csv") or f.endswith(".parquet"))]
    rd = lambda f: pd.read_parquet(f"{od}/{f}") if f.endswith(".parquet") else pd.read_csv(f"{od}/{f}")
    real_f = [f for f in de if "real" in f.lower()][0]; pred_f = [f for f in de if "pred" in f.lower()][0]
    Rt, Pt = rd(real_f), rd(pred_f)
    tcol = "target" if "target" in Rt.columns else Rt.columns[0]
    gi = {g: i for i, g in enumerate(E.S.GENES)}; ti = {t: i for i, t in enumerate(targets)}
    rows = []
    for t, rg in Rt.groupby(tcol):
        if t not in ti: continue
        rg = rg[rg.feature != t]; pg = Pt[(Pt[tcol] == t) & (Pt.feature != t)]
        Rs = rg[rg.p_adj < 0.05]; Rset = dict(zip(Rs.feature, np.sign(Rs.log2_fold_change)))
        Ps = pg[pg.p_adj < 0.05]; Pset = dict(zip(Ps.feature, np.sign(Ps.log2_fold_change)))
        gated = [g for g in rg.feature if g in gi]
        lf = full[ti[t]]; score = np.array([abs(lf[gi[g]]) for g in gated])
        K = len(Rset)
        if K == 0:
            rows.append({"target": t, "R": 0, "called": len(Pset)}); continue
        top = [gated[j] for j in np.argsort(-score)[:K]]
        hit = [g for g in top if g in Rset]
        dir_ok = np.mean([np.sign(lf[gi[g]]) == Rset[g] for g in hit]) if hit else np.nan
        inter = [g for g in Pset if g in Rset]
        rows.append({"target": t, "R": K, "called": len(Pset), "called_precision": len(inter) / max(len(Pset), 1),
                     "called_recall": len(inter) / K, "called_dir_ok": np.mean([Pset[g] == Rset[g] for g in inter]) if inter else np.nan,
                     "topK_h": len(hit) / K, "topK_dir_ok": dir_ok, "n_gated": len(gated)})
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/{name}_per_target.csv", index=False)
    s = {"targets": int(len(T)), "median_R": float(T.R.median()), "mean_R": float(T.R.mean()), "frac_targets_R0": float((T.R == 0).mean()),
         "median_called": float(T.called.median()), "called_precision": float(T.called_precision.mean()), "called_recall": float(T.called_recall.mean()),
         "called_dir_ok": float(T.called_dir_ok.mean()), "topK_h_mean": float(T.topK_h.mean()), "topK_h_median": float(T.topK_h.median()),
         "topK_dir_ok": float(T.topK_dir_ok.mean())}
    s["jac_if_controlled (h/(2-h), mean h)"] = s["topK_h_mean"] / (2 - s["topK_h_mean"])
    res[name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
