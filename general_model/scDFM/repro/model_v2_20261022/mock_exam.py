"""Mock exam on H1 (400 cells per target) with the official scorer: current pipeline vs the trained model (2026-10-22).

    .venv/bin/python -I repro/model_v2_20261022/mock_exam.py [subsets, default 0,1,2,3,4]        (from scDFM/)

For each subset data/mock_h1/s<k> (prepared by mock_data.py; H1 excluded from every source with VCC_EXCLUDE=H1):
  A "current"  tools/final_pipeline.sh, the configuration of the first D/E/F entry gm_final_v1
               (RENDER=g1 TOPFRAC=0.02 CELLAMP=1.5 CELLAGREE=1 TWPOW=0.5 ENTRYAGREE=1), NOPACK -> cells h5ad
  B "trained"  the same pipeline and settings, but the trans part comes from AdaptNet (trained once without H1, seed 0)
               instead of the consensus: own + cis from A's export; AdaptNet trans (own/cis genes zeroed, log2) rescaled
               to A's global trans norm for the cells input and for the totals input (the totals are then scaled to the
               scorer's budget = claim by the pipeline anyway), passed in with PRECOMP.
Both are scored by cell-eval2 (vcc2026 preset, pdex) against the subset's real.h5ad and scaled with its bundle
(context-mean baseline). Results: repro/model_v2_20261022/mock/<variant>_s<k>.json and a summary line per subset.
"""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

ROOT = "/home/zxc29051617/VCC"; HERE = os.path.dirname(os.path.abspath(__file__)); EFF = f"{ROOT}/data/stage3/effects"
CE2 = f"{ROOT}/evalenv/bin/cell-eval2"
COMMON = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting", "--input-type", "counts", "--set", "de.backend=pdex"]
ENV = {**os.environ, **{k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS")}}
MOCK_SET = os.environ.get("MOCK_SET", "h1")         # h1 (mock_data.py) or hct116 / kolf200 (mock_data_loco.py)
EXCL, FOLD = {"h1": ("H1", "H1"), "hct116": ("HCT116,HEK293T", "XAtlas"), "kolf200": ("FengNF,FengF,KOLF", "iPSC")}[MOCK_SET]
PFX = "" if MOCK_SET == "h1" else f"{MOCK_SET}_"      # result / effect file prefix for the other sets
PIPE_ENV = {**os.environ, "VCC_EXCLUDE": EXCL, "RENDER": "g1", "TOPFRAC": "0.02", "CELLAMP": "1.5", "CELLAGREE": "1", "TWPOW": "0.5",
            "ENTRYAGREE": "1", "NOPACK": "1"}
MEMBERS = ["pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae", "de_wilcoxon_direction_fidelity_yield_raw",
           "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard"]
os.makedirs(f"{HERE}/mock", exist_ok=True)
log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def bundle(d):
    if os.path.exists(f"{d}/bundle"): return
    subprocess.run([CE2, "baseline", "-ar", f"{d}/real.h5ad", "-o", f"{d}/baseline", "--save-pred", f"{d}/baseline_pred.h5ad", *COMMON],
                   check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([CE2, "prep-real-bundle", "--real", f"{d}/real.h5ad", "--baseline", f"{d}/baseline_pred.h5ad", "-o", f"{d}/bundle", "--force", *COMMON],
                   check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pipeline(d, name, precomp=None):
    env = dict(PIPE_ENV)
    if precomp: env["PRECOMP"] = precomp
    r = subprocess.run(["bash", f"{ROOT}/tools/final_pipeline.sh", d, name, "H"], env=env, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(f"pipeline {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def score(d, name):
    pred = f"{ROOT}/vccsub/out/{name}.h5ad"; P = ad.read_h5ad(pred); C = ad.read_h5ad(f"{d}/context_H.h5ad")
    Q = ad.AnnData(sp.vstack([sp.csr_matrix(P.X), sp.csr_matrix(C.X)]).tocsr(),
                   obs=pd.DataFrame({"target_gene": list(P.obs["target_gene"].astype(str)) + ["non-targeting"] * C.n_obs},
                                    index=[f"p{i}" for i in range(P.n_obs + C.n_obs)]), var=pd.DataFrame(index=P.var_names))
    qf = f"{ROOT}/vccsub/out/{name}_scoring.h5ad"; Q.write_h5ad(qf); del P, Q
    out = f"{HERE}/mock/{name}_ce2"
    extra = ["--write-degenes"] if os.environ.get("WRITE_DE") == "1" else []       # de_real / de_pred parquet for diagnosis
    subprocess.run([CE2, "run", "-ap", qf, "-ar", f"{d}/real.h5ad", "-o", out, *COMMON, *extra], check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    agg = [f for f in os.listdir(out) if f.endswith("agg_results.csv")][0]
    subprocess.run([CE2, "score", "--user-agg", f"{out}/{agg}", "--real-bundle", f"{d}/bundle", "-o", f"{out}/score.csv"], check=True, env=ENV,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = pd.read_csv(f"{out}/{agg}").set_index("statistic").loc["mean"]
    sc = pd.read_csv(f"{out}/score.csv")
    res = {"raw": {m: float(raw[m]) for m in MEMBERS if m in raw}, "scaled": sc.to_dict(orient="records")}
    json.dump(res, open(f"{HERE}/mock/{name}.json", "w"), indent=1)
    os.remove(qf)
    if os.environ.get("KEEP") != "1": os.remove(pred)                          # KEEP=1 keeps the predicted cells
    return res


def adaptnet_effects(model, d, k, cells_from_a=False, cells_blend=0.0):
    """PRECOMP files for variant B from A's own + cis and AdaptNet's trans. cells_from_a: the cells input stays A's
    (consensus) and only the totals input comes from AdaptNet (hybrid)."""
    import adaptnet as AN
    a = f"{EFF}/mock_{PFX}current_s{k}"; zc = np.load(f"{a}.npz", allow_pickle=True); zo = np.load(f"{a}_oc.npz", allow_pickle=True); zd = np.load(f"{a}_d200.npz", allow_pickle=True)
    targets = [str(t) for t in zc["targets"]]; oc = zo["lfc"].astype(np.float64)
    C = ad.read_h5ad(f"{d}/context_H.h5ad"); s = np.asarray(sp.csr_matrix(C.X).sum(0)).ravel(); ctrl = np.log1p(5e4 * s / s.sum())
    T = np.mean([AN.predict_ctrl(m, ctrl, targets).astype(np.float64) for m in (model if isinstance(model, list) else [model])], 0) / np.log(2)
    G = B.genome()
    for i, p in enumerate(targets):
        g, _ = G.cis(p); T[i, g] = 0.0
        if p in G.idx: T[i, G.idx[p]] = 0.0
    pre = f"{EFF}/mock_{PFX}anpre_s{k}"
    for path, src in ((f"{pre}.npz", zc), (f"{pre}_d200.npz", zd)):
        if cells_from_a and path == f"{pre}.npz":
            np.savez_compressed(path, targets=np.array(targets), lfc=zc["lfc"]); continue
        if cells_blend > 0 and path == f"{pre}.npz":       # cells: (1 - w) x consensus trans + w x AdaptNet trans, same size
            tr = zc["lfc"].astype(np.float64) - oc; kappa = np.linalg.norm(tr) / max(np.linalg.norm(T), 1e-12)
            np.savez_compressed(path, targets=np.array(targets), lfc=(oc + (1 - cells_blend) * tr + cells_blend * kappa * T).astype(np.float32)); continue
        tr = src["lfc"].astype(np.float64) - oc; kappa = np.linalg.norm(tr) / max(np.linalg.norm(T), 1e-12)
        np.savez_compressed(path, targets=np.array(targets), lfc=(oc + kappa * T).astype(np.float32))
    np.savez_compressed(f"{pre}_oc.npz", targets=np.array(targets), lfc=oc.astype(np.float32))
    return pre


def overall(res):
    rec = {r.get("metric", r.get("member", "")): r for r in res["scaled"]}
    vals = [float(r.get("score", r.get("scaled", np.nan))) for r in res["scaled"] if str(r.get("metric", r.get("member", ""))) != "overall"]
    return float(np.nanmean(vals)) if vals else float("nan")


# variant B (2026-10-22 late): AdaptNet with the full detail (beta 1) and minimal processing -- no panel shared removal
# (it costs the trained model pds on the 7-fold benchmark), no sharpening, no target or entry weights, no cell agreement
# factor; control axes, budget = claim and the top-2 % rendering as in A. Empty values switch a pipeline option off.
B_NAME = sys.argv[2] if len(sys.argv) > 2 else "trained_b1min"
B_ENV = {"SHAREDK": "0", "GAMMA": "1", "TWPOW": "", "ENTRYAGREE": "", "CELLAGREE": ""}
if len(sys.argv) > 3: B_ENV.update(dict(kv.split("=", 1) for kv in sys.argv[3].split(";") if kv))   # e.g. "CELLAGREE=1"
CELLS_FROM_A = B_ENV.pop("CELLS_FROM_A", "") == "1"
CELLS_BLEND = float(B_ENV.pop("CELLS_BLEND", "0") or 0)


def pipeline_env(d, name, precomp, extra):
    env = dict(PIPE_ENV); env.update(extra); env["PRECOMP"] = precomp
    r = subprocess.run(["bash", f"{ROOT}/tools/final_pipeline.sh", d, name, "H"], env=env, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(f"pipeline {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


if __name__ == "__main__":
    subsets = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4").split(",")]
    import adaptnet as AN
    model = None
    for k in subsets:
        d = f"{ROOT}/data/mock_{MOCK_SET}/s{k}"; t0 = time.time()
        bundle(d); log(f"s{k}: bundle ready")
        if os.path.exists(f"{HERE}/mock/mock_{PFX}current_s{k}.json") and os.path.exists(f"{EFF}/mock_{PFX}current_s{k}_oc.npz"):
            A = json.load(open(f"{HERE}/mock/mock_{PFX}current_s{k}.json")); log(f"s{k}: current (cached)")
        else:
            pipeline(d, f"mock_{PFX}current_s{k}"); A = score(d, f"mock_{PFX}current_s{k}"); log(f"s{k}: current scored ({time.time() - t0:.0f}s)")
        if model is None:
            train = [c for c in B.CONTEXTS if c not in B.FOLDS[FOLD]]; model = []
            for sd in [int(x) for x in os.environ.get("SEEDS", "0").split(",")]:      # SEEDS=0,1,2: seed ensemble
                model.append(AN.train(train, steps=1500, k=200, nce=1.0, seed=sd, log=False, beta_fix=1.0, store=model[0][1] if model else None))
        pre = adaptnet_effects(model, d, k, cells_from_a=CELLS_FROM_A, cells_blend=CELLS_BLEND)
        pipeline_env(d, f"mock_{PFX}{B_NAME}_s{k}", pre, B_ENV); Bres = score(d, f"mock_{PFX}{B_NAME}_s{k}"); log(f"s{k}: {B_NAME} scored ({time.time() - t0:.0f}s)")
        line = " | ".join(f"{m.split('_')[-1] if 'raw' in m else m.split('_')[0]} {A['raw'].get(m, np.nan):.4f}->{Bres['raw'].get(m, np.nan):.4f}" for m in MEMBERS)
        log(f"s{k} raw current->trained: {line}")
