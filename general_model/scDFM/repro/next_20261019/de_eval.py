"""Cell-level (DE) variants with the official scorer on the hold-outs, totals held fixed (2026-10-20).

    python de_eval.py [sets ...] --variants D0 D1 ...          (scDFM venv; serial; resumable; logs to run.log)

Why: on A/B/C the context-mean baseline already calls DE genes at chance direction (fid baseline 0.513), so the many
chance-level calls of the alpha-3 cells score ~0 there (F9 raw fid 0.517), while the local baselines call almost
nothing (local fid looks good). fid_yield_raw = k / max(n_pred, N_conf) rewards calling about as many genes as the
reference with correct signs; reach ranks the reference's significant genes by OUR significance.
Totals for every variant: own + cis + 0.6 x trans (depth-weighted, shared + control axes removed, gamma 3), matched to
the base trans norm (pds / mse identical across variants by construction). Cells:
  D0  own + cis + equal-weight trans x3 (= V1, the cells of every upload except F8/F9)
  D1  own + cis + depth-weighted trans x3
  D2  D1 with trans kept only on each target's top-300 genes by agreement x |t| (others 0); D3 top-1000
  D4  own + cis + depth-weighted trans x1.5 (fewer calls by smaller shifts)
  D5 / D6  D0 cells (equal-weight trans x3, which over-call ~3.6x on H1) with trans kept only on each target's top-300 /
      top-1000 genes by agreement x |t_eq| (D1-D4 use the smaller depth-weighted magnitudes, which under-call)
  D14  own + cis + 1.5 x trans x (|t_p| / median): strong predicted targets (likely many reference DE genes) get larger shifts
  D12 / D13  own + cis + 1.5 x trans x (agreement / its median)^1 / ^2 per (target, gene): genes whose sources agree in sign
      get larger shifts, so our significance order follows confidence (reach)
  D10 / D11  own + cis + trans + 0.5 / 1.0 x its target-specific part (panel mean and top-10 shared components kept x1)
  D7 / D8 / D9  D0 cells with the trans amplitude x0.75 / x1.5 / x2 (alpha 2.25 / 4.5 / 6): how fid moves with the over-calling level
  suffix f / g (e.g. D0f, D0g): the dual renderer's added counts go to the top 2 % / 0.5 % of a target's cells instead of 10 %
      (less distortion of the per-cell distribution that the DE tests see; totals unchanged)
Same control draws and seeds as render_eval (seed 0, 200 cells per target). Reports raw fid, reach, jac, nmae.
"""
import json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import render_eval as E, sigexp as S, weights_screen as W

HERE = E.HERE; SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"


def parts(name, C, lib, targets, oc):
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; Cc = S.load_cache(train); ti = {t: i for i, t in enumerate(targets)}
    t_eq = W.trans(Cc, train, targets, 0, W.WEIGHTS["equal"]); t_d = W.trans(Cc, train, targets, 0, W.WEIGHTS["depth200"])
    n = np.zeros_like(oc); s1 = np.zeros_like(oc); pos = np.zeros_like(oc)
    for c in train:
        Y = Cc[c]["Ysrc"]
        for j, t in enumerate(Cc[c]["targets"]):
            i = ti.get(t)
            if i is None: continue
            y = Y[j]; ok = np.isfinite(y); yz = np.where(ok, y, 0); n[i] += ok; s1[i] += yz; pos[i] += yz > 0
    mean = s1 / np.maximum(n, 1); agree = np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))
    V = E.control_axes(C, lib, 50).astype(np.float64)
    t5 = E.remove_shared(t_d.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
    t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2)[:, None]
    base = E.remove_shared((t_eq).astype(np.float32)).astype(np.float64); base = base - (base @ V.T) @ V
    tot = oc + 0.6 * t8 * (np.linalg.norm(base) / np.linalg.norm(t8))
    return t_eq, t_d, agree, tot


def cells_for(v, oc, t_eq, t_d, agree):
    v = v.rstrip("fg")
    if v == "D0": return oc + t_eq
    if v == "D1": return oc + t_d
    if v in ("D2", "D3"):
        N = 300 if v == "D2" else 1000
        key = np.abs(t_d) * agree; thr = -np.sort(-key, axis=1)[:, N - 1:N]
        return oc + np.where(key >= thr, t_d, 0.0)
    if v == "D4": return oc + 0.5 * t_d
    if v == "D7": return oc + 0.75 * t_eq
    if v == "D8": return oc + 1.5 * t_eq
    if v == "D9": return oc + 2.0 * t_eq
    if v == "D14":                                      # cells x4.5 with per-target sharpening (strong targets get more calls)
        nr = np.linalg.norm(E.remove_shared(t_eq.astype(np.float32)).astype(np.float64), axis=1); f = nr / np.median(nr)
        return oc + 1.5 * t_eq * f[:, None]
    if v in ("D12", "D13"):                             # cells x4.5 with per-entry sign agreement as a confidence factor
        g = np.clip(agree / np.median(agree[np.abs(t_eq) > 0]), 0, None) ** (1 if v == "D12" else 2)
        return oc + 1.5 * t_eq * g
    if v in ("D10", "D11"):                             # amplify only the target-specific part (shared part kept x1)
        spec = E.remove_shared(t_eq.astype(np.float32)).astype(np.float64)
        return oc + t_eq + (0.5 if v == "D10" else 1.0) * spec
    if v in ("D5", "D6"):
        N = 300 if v == "D5" else 1000
        key = np.abs(t_eq) * agree; thr = -np.sort(-key, axis=1)[:, N - 1:N]
        return oc + np.where(key >= thr, t_eq, 0.0)
    raise ValueError(v)


def main():
    args = sys.argv[1:]; seed = 0
    if "--seed" in args:
        k = args.index("--seed"); seed = int(args[k + 1]); args = args[:k] + args[k + 2:]
    vi = args.index("--variants") if "--variants" in args else len(args)
    names = args[:vi] or ["H1", "KOLF200"]; variants = args[vi + 1:] or ["D0", "D1", "D2", "D3", "D4"]
    E.log(f"de_eval start: sets {names}, variants {variants} (totals fixed; raw fid/reach/jac/nmae compared)")
    for name in names:
        D, R, C, lib, rate, ref_mean = E.load_set(name); targets, oc, full, shuf = E.components(name)
        t_eq, t_d, agree, tot = parts(name, C, lib, targets, oc.astype(np.float64))
        for v in variants:
            tag = f"{name}_{v}_seed{seed}"; out = f"{HERE}/scores_de/{tag}"
            if os.path.exists(f"{out}/score.csv"): continue
            while E.avail_gb() < 5: time.sleep(60)
            t0 = time.time(); os.makedirs(f"{HERE}/pred", exist_ok=True); pf = f"{HERE}/pred/{tag}.h5ad"
            import vcc.render as VR
            VR.TOP_FRAC = 0.02 if v.endswith("f") else (0.005 if v.endswith("g") else 0.1)
            P = E._generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, cells_for(v, oc, t_eq, t_d, agree).astype(np.float32), tot.astype(np.float32))
            P.write_h5ad(pf); del P
            shutil.rmtree(out, ignore_errors=True)
            subprocess.run([E.CE2, "run", "-ap", pf, "-ar", f"{D}/real.h5ad", "-o", out, *E.COMMON, "--write-degenes"], check=True, env=E.ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dp = pd.read_parquet(f"{out}/de_pred.parquet", columns=["target", "feature", "p_adj"]); dr = pd.read_parquet(f"{out}/de_real.parquet", columns=["target", "feature", "p_adj"])
            npred = dp[(dp.p_adj < 0.05) & (dp.feature != dp.target)].groupby("target").size().reindex(targets, fill_value=0)
            nconf = dr[(dr.p_adj < 0.05) & (dr.feature != dr.target)].groupby("target").size().reindex(targets, fill_value=0)
            calls = f"n_pred median {npred.median():.0f} (IQR {npred.quantile(.25):.0f}-{npred.quantile(.75):.0f}) vs N_conf median {nconf.median():.0f}"
            for f in ("de_pred.parquet", "de_real.parquet"): os.remove(f"{out}/{f}")
            agg = [f for f in os.listdir(out) if f.endswith("agg_results.csv")][0]
            subprocess.run([E.CE2, "score", "--user-agg", f"{out}/{agg}", "--real-bundle", f"{D}/bundle", "-o", f"{out}/score.csv"], check=True, env=E.ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(pf)
            raw = pd.read_csv(f"{out}/{agg}").set_index("statistic").loc["mean"]
            E.log(f"de_eval {tag}: raw fid {raw['de_wilcoxon_direction_fidelity_yield_raw']:.4f} reach {raw['de_wilcoxon_direction_reach_raw']:.4f} "
                  f"jac {raw['de_wilcoxon_sig_jaccard']:.4f} nmae {raw['de_wilcoxon_lfc_nmae']:.4f} | {calls} | pds {raw['pds_cosine']:.4f} mse {raw['expr_mse_unbiased_capped_norm']:.3f} ({time.time() - t0:.0f}s)")
    E.log("de_eval finished")


if __name__ == "__main__":
    main()
