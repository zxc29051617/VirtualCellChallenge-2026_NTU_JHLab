"""Option A done properly (2026-10-19): exact-total rendering and separate bulk / cell-level amplitudes.

    python render_eval.py check            # renderer sanity checks on a few targets (no scoring)
    python render_eval.py run [set ...]    # generation + official scoring, serial, resumable

Same model and same control draws as the SIGnature experiment's group A runs (repro/signature_20261013, uniform source
weights, trans amplitude 3, own -1.9 ln, cis kernel), only the rendering changes:
  V0   current renderer, trans x3                     (= signature_20261013 scores <set>_A_seed<k>, reused)
  V1   exact gene totals, trans x3 for cells and bulk
  V2   dual: cells trans x3, bulk trans x1 (own + cis at full strength in both)
  V2s  V2 with each target's trans part taken from another target (the A_shuf derangement)
  V5   V1 (exact totals) with the mean and the top-10 principal components of the panel's trans part removed
       (prediction-only; added 2026-10-19 17:40 after a pds screen); V5s its pairing control (same removal applied
       to the shuffled trans rows).
  V6   cells as V1 (full trans), gene totals as V5 (shared components removed), render mode dual; V6s its pairing
       control (added 2026-10-19 18:45 after the V5 result; truth-free check: keeps 75-81 % of V1's cell-level change
       on the genes most changed by the removal, V5 51-62 %).
  V7a/V7b  V6 with the gene totals' whole log-change (own + cis + shared-removed trans) scaled by 0.5 / 0.25 (pds is
       unchanged by a per-target scale; aimed at the mse member); V7as = V7a pairing control. 1 seed (totals are
       deterministic; seeds only move the DE members). Added 2026-10-19 21:55.
  V8   V6 with the top-50 principal axes of the context's own control cells (log1p CP10k) also removed from the totals'
       trans part (cells as V1, render mode dual); V8s its pairing control. Added 2026-10-20 00:40 after the bulk-only pds
       screen (pds_screen2.py: +0.009 / +0.015 / +0.031 on H1 / KOLF200 / HCT116).
  V9   V8 totals scaled by 0.5 (the V7a shrink combined with V8); V9s its pairing control. 1 seed, a readiness check for
       the leaderboard decision after V7a (2026-10-20 02:15).
  V10  totals as V6; cells with the shared trans part (panel mean + top-10 components) doubled -- a cells-only change aimed
       at the DE members (pds and mse unchanged by construction). 1 seed screen (2026-10-20 02:45).
  F10 / F10b  cells V10; totals = own + cis unscaled + V8 trans part x0.5 / x0.35 (only the transferred part shrunk);
       F10s pairing control (2026-10-20 09:10, after the V7a leaderboard result).
  F8 / F9  final candidates: cells as V10, totals as V8 (F8) or V9 (F9); F8s pairing control (2026-10-20 03:30).
  V3   as V2 but the bulk adjustment is made in the cells with the largest libraries (render mode dual_lib);
       V3s its pairing control. V3/V3s were added (2026-10-19 15:00) before any V1/V2 score was read.
Control indices per target use the same stream as signature_20261013 ([seed, i, 0]); thinning/Poisson the same
stream ([seed, i, 1]); the total fix its own stream. 200 cells per target, scorer cell-eval2 0.18.0 vcc2026 + pdex,
existing fixed bundles. Pre-set pass rule (per hold-out; needed in >= 2 of 3, see analyze_render.py):
raw panel mse_norm of V2 below V0 in all 3 seeds; per-target pds V2 - V0 mean >= -0.01 and 95% CI low >= -0.02;
V2 avg above V2s in all 3 seeds and per-target pds V2 - V2s CI above 0.
"""
import json, os, shutil, subprocess, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/home/zxc29051617/VCC"; SIG = f"{ROOT}/scDFM/repro/signature_20261013"
sys.path.insert(0, f"{ROOT}/scDFM"); sys.path.insert(0, SIG)
import sigexp as S
from vcc.gm import Genome
from vcc.render import render

CE2 = f"{ROOT}/evalenv/bin/cell-eval2"; N_PRED = 200; SEEDS = (0, 1, 2); VARIANTS = ["V1", "V2", "V2s"]; BULK_TRANS = 1.0 / 3.0
COMMON = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting", "--input-type", "counts", "--set", "de.backend=pdex"]
ENV = {**os.environ, **{k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS")}}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"; print(line, flush=True); open(f"{HERE}/run.log", "a").write(line + "\n")


def avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 2**20


def components(name):
    """targets, own+cis lfc, full lfc (trans x3), shuffled full lfc -- all log2, from the signature experiment."""
    o = S.OUTER[name]; W = json.load(open(f"{SIG}/weights_{name}.json")); train = W["train"]
    za = np.load(f"{SIG}/effects/{name}_A.npz", allow_pickle=True); zs = np.load(f"{SIG}/effects/{name}_A_shuf.npz", allow_pickle=True)
    targets = list(za["targets"]); assert list(zs["targets"]) == targets
    oc = S.build_effects(S.load_cache(train), train, {c: 1.0 for c in train}, targets, Genome(S.GENES), use_trans=False)
    return targets, oc.astype(np.float32), za["lfc"].astype(np.float32), zs["lfc"].astype(np.float32)


PC_K = 10


def remove_shared(trans, k=PC_K):
    """Remove the mean and the top-k principal components of the panel's trans part (rows = targets). Uses only
    predictions; at submission time it is computed across the 300 challenge targets of each context."""
    t = trans.astype(np.float64); t = t - t.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(t, full_matrices=False)
    return (t - (t @ vt[:k].T) @ vt[:k]).astype(np.float32)


def control_axes(C, lib, m, n=2000, seed=0):
    """Top-m principal axes (gene space) of the context's own control cells, log1p CP10k, centred. No truth used."""
    idx = np.random.default_rng(seed).choice(C.shape[0], min(n, C.shape[0]), replace=False)
    X = C[idx].multiply(1e4 / lib[idx][:, None]).tocsr(); X.data = np.log1p(X.data); X = X.toarray(); X -= X.mean(0, keepdims=True)
    return np.linalg.svd(X, full_matrices=False)[2][:m].astype(np.float32)


def lfcs(variant, oc, full, shuf):
    if variant == "V1": return full, None, "exact"
    if variant == "V2": return full, oc + (full - oc) * BULK_TRANS, "dual"
    if variant == "V2s": return shuf, oc + (shuf - oc) * BULK_TRANS, "dual"
    if variant in ("V5", "V5s"): return (full if variant == "V5" else shuf), None, "exact"
    if variant == "V3": return full, oc + (full - oc) * BULK_TRANS, "dual_lib"
    if variant == "V3s": return shuf, oc + (shuf - oc) * BULK_TRANS, "dual_lib"
    raise ValueError(variant)


def generate(name, seed, variant, R, C, lib, rate, ref_mean, comp):
    targets, oc, full, shuf = comp
    if variant in ("V5", "V5s"):                         # shared-component removal on the whole panel, then exact totals
        full = oc + remove_shared(full - oc); shuf = oc + remove_shared(shuf - oc)
    if variant in ("F10", "F10s", "F10b"):              # cells V10; totals = own + cis at full strength + trans x0.5 (F10) / x0.35 (F10b)
        src = shuf if variant == "F10s" else full; tr = (src - oc).astype(np.float64); t5 = remove_shared(src - oc).astype(np.float64)
        cells = oc + tr + (tr - tr.mean(0, keepdims=True) - t5)
        V = control_axes(C, lib, 50).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
        tot = oc + (0.35 if variant == "F10b" else 0.5) * t8
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, cells.astype(np.float32), tot.astype(np.float32))
    if variant in ("F8", "F8s", "F9"):                  # final candidates: cells V10, totals V8 (F8) or V9 (F9)
        src = shuf if variant == "F8s" else full; tr = (src - oc).astype(np.float64); t5 = remove_shared(src - oc).astype(np.float64)
        cells = oc + tr + (tr - tr.mean(0, keepdims=True) - t5)
        V = control_axes(C, lib, 50).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
        tot = (oc + t8) * (0.5 if variant == "F9" else 1.0)
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, cells.astype(np.float32), tot.astype(np.float32))
    if variant in ("V10g05", "V10g2"):                  # V10 with the extra shared part x0.5 (cells shared x1.5) or x2 (x3)
        g = 0.5 if variant == "V10g05" else 2.0
        tr = (full - oc).astype(np.float64); t5 = remove_shared(full - oc).astype(np.float64)
        cells = oc + tr + g * (tr - tr.mean(0, keepdims=True) - t5)
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, cells.astype(np.float32), oc + t5)
    if variant in ("V10", "V10s"):                       # cells: shared trans components doubled; totals: V6
        src = full if variant == "V10" else shuf; tr = (src - oc).astype(np.float64); t5 = remove_shared(src - oc).astype(np.float64)
        cells = oc + tr + (tr - tr.mean(0, keepdims=True) - t5)          # shared part (mean + top-10 PCs) counted twice
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, cells.astype(np.float32), oc + t5)
    if variant in ("V9", "V9s"):                         # V8 totals scaled by 0.5 (V7a x V8)
        src = full if variant == "V9" else shuf; V = control_axes(C, lib, 50)
        t5 = remove_shared(src - oc); t5 = t5 - (t5 @ V.T) @ V
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, src, 0.5 * (oc + t5))
    if variant in ("V8", "V8s"):                         # V6 + the context's top-50 control-cell axes removed from the totals' trans
        src = full if variant == "V8" else shuf; V = control_axes(C, lib, 50)
        t5 = remove_shared(src - oc); t5 = t5 - (t5 @ V.T) @ V
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, src, oc + t5)
    if variant in ("V7a", "V7b", "V7as"):                # V6 with the whole gene-total change shrunk (pds is scale-free)
        sc = 0.25 if variant == "V7b" else 0.5; src = shuf if variant == "V7as" else full
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, src, sc * (oc + remove_shared(src - oc)))
    if variant in ("V6", "V6s"):                         # cells follow V1, gene totals follow V5
        src = full if variant == "V6" else shuf; bulk5 = oc + remove_shared(src - oc)
        return _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, src, bulk5)
    blocks, labels = [], []
    for i, t in enumerate(targets):
        draw = np.random.default_rng([seed, i, 0]).choice(C.shape[0], N_PRED, replace=False); rng = np.random.default_rng([seed, i, 1])
        lc, lb, mode = lfcs(variant, oc[i], full[i], shuf[i])
        blocks.append(render(C[draw], lib[draw], rate, ref_mean, lc, rng, mode=mode, lfc_bulk=lb, fix_seed=seed * 1_000_003 + i))
        labels += [t] * N_PRED
    P = ad.AnnData(sp.vstack(blocks + [C]).tocsr(), obs=pd.DataFrame({"target_gene": labels + ["non-targeting"] * C.shape[0]}), var=pd.DataFrame(index=R.var_names))
    P.obs_names = [f"p{i}" for i in range(P.n_obs)]
    return P


def _generate_pairs(name, seed, R, C, lib, rate, ref_mean, targets, cells_lfc, bulk_lfc):
    blocks, labels = [], []
    for i, t in enumerate(targets):
        draw = np.random.default_rng([seed, i, 0]).choice(C.shape[0], N_PRED, replace=False); rng = np.random.default_rng([seed, i, 1])
        blocks.append(render(C[draw], lib[draw], rate, ref_mean, cells_lfc[i], rng, mode="dual", lfc_bulk=bulk_lfc[i], fix_seed=seed * 1_000_003 + i))
        labels += [t] * N_PRED
    P = ad.AnnData(sp.vstack(blocks + [C]).tocsr(), obs=pd.DataFrame({"target_gene": labels + ["non-targeting"] * C.shape[0]}), var=pd.DataFrame(index=R.var_names))
    P.obs_names = [f"p{i}" for i in range(P.n_obs)]
    return P


def load_set(name):
    D = f"{ROOT}/data/stage3/loco/{S.OUTER[name]['loco']}"
    R = ad.read_h5ad(f"{D}/real.h5ad"); C = sp.csr_matrix(R[R.obs.target_gene == "non-targeting"].X)
    lib = np.asarray(C.sum(1)).ravel(); rate = np.asarray(C.sum(0)).ravel() / lib.sum(); ref_mean = np.asarray(C.sum(0)).ravel() / C.shape[0]
    return D, R, C, lib, rate, ref_mean


def check():
    for name in ["H1"]:
        D, R, C, lib, rate, ref_mean = load_set(name); comp = components(name); targets, oc, full, shuf = comp
        assert np.allclose(oc + (full - oc), full)
        X = sp.csr_matrix(R.X); lab = R.obs.target_gene.astype(str).values
        cn = X[lab == "non-targeting"]; cn = cn.multiply(1e4 / np.asarray(cn.sum(1))).tocsr(); cn.data = np.log1p(cn.data)
        med_ctrl = np.asarray(cn.mean(0)).ravel()
        for i in [0, 1, 2]:
            draw = np.random.default_rng([0, i, 0]).choice(C.shape[0], N_PRED, replace=False)
            out = {}
            for v in ["V0", "V1", "V2", "V3"]:
                if v == "V0": lc, lb, mode = full[i], None, "current"
                else: lc, lb, mode = lfcs(v, oc[i], full[i], shuf[i])
                Y = render(C[draw], lib[draw], rate, ref_mean, lc, np.random.default_rng([0, i, 1]), mode=mode, lfc_bulk=lb, fix_seed=i)
                tot = np.asarray(Y.sum(0)).ravel()
                fb = 2.0 ** np.clip(lb if lb is not None else lc, -4, 4); T = np.rint(N_PRED * ref_mean * fb)
                yn = Y.multiply(1e4 / np.asarray(Y.sum(1))).tocsr(); yn.data = np.log1p(yn.data)
                gm = np.asarray(yn.mean(0)).ravel()
                tr = np.abs(full[i] - oc[i]) > 0.3                                  # genes with a sizeable trans change
                out[v] = {"exact_totals": bool(np.all(tot == T)) if mode != "current" else None,
                          "bulk_log2fc_on_trans_genes": float(np.median(np.log2((tot[tr] + 1) / (N_PRED * ref_mean[tr] + 1)) * np.sign(full[i][tr]))),
                          "cell_meanlog_shift_on_trans_genes": float(np.median((gm[tr] - med_ctrl[tr]) * np.sign(full[i][tr])))}
            log(f"check {name} target {targets[i]}: " + json.dumps(out))


def run(names, variants=VARIANTS, seeds=SEEDS):
    total = len(names) * len(seeds) * len(variants); done = 0
    for seed in seeds:
        for name in names:
            D, R, C, lib, rate, ref_mean = load_set(name); comp = components(name)
            for v in variants:
                tag = f"{name}_{v}_seed{seed}"; out = f"{HERE}/scores/{tag}"
                if os.path.exists(f"{out}/score.csv"): done += 1; continue
                while avail_gb() < 5: log(f"waiting for memory ({avail_gb():.1f} GB available)"); time.sleep(60)
                t0 = time.time(); os.makedirs(f"{HERE}/pred", exist_ok=True); pf = f"{HERE}/pred/{tag}.h5ad"
                P = generate(name, seed, v, R, C, lib, rate, ref_mean, comp); P.write_h5ad(pf); tg = time.time() - t0
                shutil.rmtree(out, ignore_errors=True)
                subprocess.run([CE2, "run", "-ap", pf, "-ar", f"{D}/real.h5ad", "-o", out, *COMMON], check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                agg = [f for f in os.listdir(out) if f.endswith("agg_results.csv")][0]
                subprocess.run([CE2, "score", "--user-agg", f"{out}/{agg}", "--real-bundle", f"{D}/bundle", "-o", f"{out}/score.csv"], check=True, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(pf); done += 1
                sc = pd.read_csv(f"{out}/score.csv").set_index("metric")["from_replicate"]
                raw = pd.read_csv(f"{out}/{agg}").set_index("statistic").loc["mean"]
                log(f"score {done}/{total}: {tag} avg {sc['avg_score']:.4f} raw mse {raw['expr_mse_unbiased_capped_norm']:.3f} raw pds {raw['pds_cosine']:.4f} "
                    f"(gen {tg:.0f}s, total {time.time() - t0:.0f}s)")
    log(f"render scoring finished ({','.join(variants)})")


if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        args = sys.argv[2:]; vs = VARIANTS
        sd = SEEDS
        if "--seeds" in args: i = args.index("--seeds"); sd = tuple(int(x) for x in args[i + 1].split(",")); args = args[:i] + args[i + 2:]
        if "--variants" in args: i = args.index("--variants"); vs = args[i + 1:]; args = args[:i]
        run(args or ["H1", "KOLF200", "HCT116"], vs, sd)
