"""Small held-out-KOLF pilot: control resample vs own+cis vs gm plain vs shuffled trans, 3 generation seeds.

Deviation from the request (documented in status.json / SUMMARY.md): only 7 of the 279 KOLFbig targets have >= 400
real cells in data/stage3/kolf_KOLF.h5ad, so the 12 targets are drawn (seed 20261011) from the 22 targets with >= 350
real cells, and the truth set uses 350 cells per target. Predictions use 400 cells per target.

    python pilot.py prep | gen | score | analyze      (run from this directory, scDFM venv; score uses evalenv)
"""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/zxc29051617/VCC"
SEED, N_TARGETS, MIN_REAL, N_TRUTH, N_CTRL, N_PRED, GEN_SEEDS = 20261011, 12, 350, 350, 4000, 400, (0, 1, 2)
EFFECTS = {"own_cis": f"{ROOT}/data/stage3/effects/KOLFbig_gm_owncis.npz",
           "gm_plain": f"{ROOT}/data/stage3/effects/KOLFbig_gm_l3k1_a3.0.npz",
           "shuffled": f"{ROOT}/data/stage3/effects/KOLFbig_gm_shuf.npz"}
GROUPS = ["control_resample", "own_cis", "gm_plain", "shuffled"]
# strict donor control: training also excludes the HipSci/Feng iPSC screens (donor lines kolf_2/kolf_3)
STRICT_EXCLUDE = ["KOLF", "FengNF", "FengF"]
EFFECTS.update({"strict_plain": f"{HERE}/effects/strict_gm_plain.npz", "strict_shuffled": f"{HERE}/effects/strict_shuffled.npz"})
ALL_GROUPS = GROUPS + ["strict_plain", "strict_shuffled"]
CE2 = f"{ROOT}/evalenv/bin/cell-eval2"
COMMON = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting", "--input-type", "counts",
          "--set", "de.backend=pdex"]
THREADS = {k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                             "POLARS_MAX_THREADS", "RAYON_NUM_THREADS")}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    open(f"{HERE}/run.log", "a").write(line + "\n")


def status(**kw):
    p = f"{HERE}/status.json"; s = json.load(open(p)); s.update(kw); s["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(s, open(p, "w"), indent=1)


def prep():
    A = ad.read_h5ad(f"{ROOT}/data/stage3/kolf_KOLF.h5ad")
    tg = A.obs.target_gene.astype(str).values
    pool = json.load(open(f"{ROOT}/data/stage3/loco/KOLFbig/real.json"))["targets"]
    counts = pd.Series(tg).value_counts()
    eligible = sorted(t for t in pool if counts.get(t, 0) >= MIN_REAL)
    n400 = sum(counts.get(t, 0) >= 400 for t in pool)
    rng = np.random.default_rng(SEED)
    targets = sorted(rng.choice(eligible, N_TARGETS, replace=False).tolist())
    log(f"prep: {n400} targets have >=400 real cells (need 12) -> drawing 12 of {len(eligible)} with >={MIN_REAL}: {targets}")
    ctrl_all = np.flatnonzero(A.obs.is_control.values)
    ctrl = np.sort(np.random.default_rng([SEED, 999999]).choice(ctrl_all, min(N_CTRL, len(ctrl_all)), replace=False))
    rows, labels = [], []
    for i, t in enumerate(targets):
        idx = np.flatnonzero(tg == t)
        pick = np.sort(np.random.default_rng([SEED, i]).choice(idx, N_TRUTH, replace=False))
        rows.append(pick); labels += [t] * N_TRUTH
    keep = np.concatenate(rows + [ctrl])
    X = sp.csr_matrix(A.X[keep]); X.eliminate_zeros()
    R = ad.AnnData(X, obs=pd.DataFrame({"target_gene": labels + ["non-targeting"] * len(ctrl)}),
                   var=pd.DataFrame(index=A.var_names))
    R.obs_names = [f"r{i}" for i in range(R.n_obs)]
    os.makedirs(f"{HERE}/data", exist_ok=True)
    R.write_h5ad(f"{HERE}/data/real.h5ad")
    man = {"seed": SEED, "source": f"{ROOT}/data/stage3/kolf_KOLF.h5ad", "target_pool": "loco/KOLFbig/real.json (279)",
           "targets_with_ge400_real_cells": int(n400), "min_real_cells_used": MIN_REAL, "eligible": eligible,
           "targets": targets, "real_cells_per_target": {t: int(counts[t]) for t in targets},
           "truth_cells_per_target": N_TRUTH, "n_ctrl": int(len(ctrl)), "pred_cells_per_target": N_PRED,
           "gen_seeds": list(GEN_SEEDS), "effects": EFFECTS}
    json.dump(man, open(f"{HERE}/manifest.json", "w"), indent=1)
    env = {**os.environ, **THREADS}
    t0 = time.time()
    subprocess.run([CE2, "baseline", "-ar", f"{HERE}/data/real.h5ad", "-o", f"{HERE}/data/baseline",
                    "--save-pred", f"{HERE}/data/baseline_pred.h5ad", *COMMON], check=True, env=env)
    subprocess.run([CE2, "prep-real-bundle", "--real", f"{HERE}/data/real.h5ad", "--baseline",
                    f"{HERE}/data/baseline_pred.h5ad", "-o", f"{HERE}/data/bundle", "--force", *COMMON], check=True, env=env)
    log(f"prep: real set {R.n_obs} cells ({N_TARGETS} x {N_TRUTH} + {len(ctrl)} NTC), bundle built in {time.time() - t0:.0f}s")


def verify(targets, eff):
    genes = None
    rep = {}
    for g, z in eff.items():
        t = list(z["targets"]); assert all(x in t for x in targets), g
    pl, sh, oc = eff["gm_plain"], eff["shuffled"], eff["own_cis"]
    assert list(pl["targets"]) == list(sh["targets"]) == list(oc["targets"])
    allt = list(pl["targets"]); gi = {g: k for k, g in enumerate(json_genes())}
    for t in targets:
        i = allt.index(t); k = gi[t]
        a, b, c = pl["lfc"][i], sh["lfc"][i], oc["lfc"][i]
        nz = np.flatnonzero(c != 0)                                   # own + cis genes of t
        rep[t] = {"own_equal_plain_shuf_owncis": bool(np.isclose(a[k], b[k]) and np.isclose(a[k], c[k])),
                  "n_own_cis_genes": int(len(nz)),
                  "corr_plain_vs_shuffled": float(np.corrcoef(a, b)[0, 1]),
                  "rows_identical_plain_shuffled": bool(np.allclose(a, b))}
    return rep


def json_genes():
    return pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist()


def strict_fit():
    """One model fitted without KOLF/FengNF/FengF; plain and shuffled rows from that SAME fit. Targets = the 279
    KOLFbig targets in real.json order, so shuffle_trans reuses the original derangement (rng(0) over 279)."""
    sys.path.insert(0, "/home/zxc29051617/VCC/scDFM")
    from vcc.contexts import load_all
    from vcc.gm import GeneralModel
    tl = json.load(open(f"{ROOT}/data/stage3/loco/KOLFbig/real.json"))["targets"]
    C = load_all(keep_var=False); train = {k: v for k, v in C.items() if k not in STRICT_EXCLUDE}
    log(f"strict_fit: training contexts {sorted(train)}")
    m = GeneralModel(per_celltype=False, lam=3.0, kcap=1.0, alpha=3.0).fit(train, targets=tl)
    os.makedirs(f"{HERE}/effects", exist_ok=True)
    for name, shuf in (("strict_gm_plain", False), ("strict_shuffled", True)):
        m.shuffle_trans = shuf
        lfc = m.predict(list(tl)) / np.log(2)
        np.savez_compressed(f"{HERE}/effects/{name}.npz", targets=np.array(tl), lfc=lfc.astype(np.float32))
    # check the derangement is the original one: original shuffled row t == original plain row of donor(t) in trans genes
    rng = np.random.default_rng(0); perm = rng.permutation(len(tl))
    while (perm == np.arange(len(tl))).any(): perm = rng.permutation(len(tl))
    orig_pl, orig_sh = np.load(EFFECTS["gm_plain"])["lfc"], np.load(EFFECTS["shuffled"])["lfc"]
    oc = np.load(EFFECTS["own_cis"])["lfc"]
    sp_, ss_ = np.load(f"{HERE}/effects/strict_gm_plain.npz")["lfc"], np.load(f"{HERE}/effects/strict_shuffled.npz")["lfc"]
    i = 0; j = perm[i]; trans_genes = (oc[i] == 0) & (oc[j] == 0)
    same_orig = bool(np.allclose(orig_sh[i][trans_genes], orig_pl[j][trans_genes], atol=1e-5))
    same_strict = bool(np.allclose(ss_[i][trans_genes], sp_[j][trans_genes], atol=1e-5))
    man = json.load(open(f"{HERE}/manifest.json"))
    man["strict"] = {"training_contexts": sorted(train), "excluded": STRICT_EXCLUDE,
                     "derangement": "numpy default_rng(0).permutation(279) re-drawn until no fixed point (vcc/gm.py)",
                     "check_original_shuffled_row0_is_plain_row_of_donor": same_orig,
                     "check_strict_shuffled_row0_is_plain_row_of_donor": same_strict,
                     "donor_evidence": "Feng 2025 cell metadata (data/external/feng2025/GenomeWideScreen_NonFitnessGenes_Cell-Metadata.tsv.gz, column Cell_Line) lists kolf_2 and kolf_3; KOLF2.1J derives from the HipSci KOLF2 donor (Pantazis et al. 2022, Cell Stem Cell)"}
    json.dump(man, open(f"{HERE}/manifest.json", "w"), indent=1)
    log(f"strict_fit: wrote strict effects; derangement checks original {same_orig}, strict {same_strict}")


def gen(groups=None):
    groups = groups or GROUPS
    man = json.load(open(f"{HERE}/manifest.json")); targets = man["targets"]
    R = ad.read_h5ad(f"{HERE}/data/real.h5ad")
    C = sp.csr_matrix(R[R.obs.target_gene == "non-targeting"].X)
    lib = np.asarray(C.sum(1)).ravel(); rate = np.asarray(C.sum(0)).ravel() / lib.sum()
    eff = {g: np.load(p, allow_pickle=True) for g, p in EFFECTS.items() if os.path.exists(p)}
    if "verify" not in man:
        man["verify"] = verify(targets, eff); json.dump(man, open(f"{HERE}/manifest.json", "w"), indent=1)
    lfc = {g: {t: np.clip(np.nan_to_num(v), -4, 4) for t, v in zip(z["targets"], z["lfc"]) if t in targets}
           for g, z in eff.items()}
    os.makedirs(f"{HERE}/pred", exist_ok=True)
    for s in GEN_SEEDS:
        draws = {t: np.random.default_rng([s, i, 0]).choice(C.shape[0], N_PRED, replace=False) for i, t in enumerate(targets)}
        json.dump({t: d.tolist() for t, d in draws.items()}, open(f"{HERE}/pred/control_indices_seed{s}.json", "w"))
        for g in groups:
            blocks, labels = [], []
            for i, t in enumerate(targets):
                draw = draws[t]; rng = np.random.default_rng([s, i, 1])
                cells = C[draw].tocsr(copy=True)
                if g != "control_resample":
                    fc = 2.0 ** lfc[g][t]
                    f = fc[cells.indices]; down = f < 1
                    d = cells.data.astype(np.int64); d[down] = rng.binomial(d[down], f[down]); cells.data = d.astype(np.float32)
                    up = np.flatnonzero(fc > 1)
                    if len(up):
                        mu = lib[draw][:, None] * rate[None, up] * (fc[up] - 1)[None, :]   # original library
                        add = sp.csr_matrix(rng.poisson(mu).astype(np.float32))
                        cells = cells + sp.csr_matrix((add.data, up[add.indices], add.indptr), shape=cells.shape)
                cells.eliminate_zeros(); blocks.append(cells); labels += [t] * N_PRED
            P = ad.AnnData(sp.vstack(blocks + [C]).tocsr(),
                           obs=pd.DataFrame({"target_gene": labels + ["non-targeting"] * C.shape[0]}),
                           var=pd.DataFrame(index=R.var_names))
            P.obs_names = [f"p{i}" for i in range(P.n_obs)]
            P.write_h5ad(f"{HERE}/pred/{g}_seed{s}.h5ad")
            log(f"gen: {g} seed {s}: {P.n_obs} cells, {P.X.nnz / P.n_obs:.0f} nnz/cell")


def score(only=None):
    env = {**os.environ, **THREADS}
    for s in GEN_SEEDS:
        for g in ALL_GROUPS:
            tag = f"{g}_seed{s}"
            if only and tag not in only: continue
            if not os.path.exists(f"{HERE}/pred/{tag}.h5ad"): continue
            out = f"{HERE}/scores/{tag}"
            if os.path.exists(f"{out}/score.csv"): continue
            t0 = time.time()
            subprocess.run([CE2, "run", "-ap", f"{HERE}/pred/{tag}.h5ad", "-ar", f"{HERE}/data/real.h5ad", "-o", out, *COMMON],
                           check=True, env=env, stdout=subprocess.DEVNULL)
            agg = [f for f in os.listdir(out) if f.endswith("agg_results.csv")][0]
            subprocess.run([CE2, "score", "--user-agg", f"{out}/{agg}", "--real-bundle", f"{HERE}/data/bundle",
                            "-o", f"{out}/score.csv"], check=True, env=env, stdout=subprocess.DEVNULL)
            avg = pd.read_csv(f"{out}/score.csv").set_index("metric").loc["avg_score", "from_replicate"]
            log(f"score: {tag} avg {avg:.4f} ({time.time() - t0:.0f}s)")
            status(step="5/7 scoring", last_scored=tag)


M = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
     "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach",
     "de_wilcoxon_sig_jaccard": "jac"}
PER_TARGET = {"pds_cosine": "pds", "de_wilcoxon_lfc_nmae": "nmae", "de_wilcoxon_direction_fidelity_yield_raw": "fid",
              "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac",
              # diagnostics only: the official mse member (capped_norm) exists only as a panel ratio of sums
              "expr_mse_unbiased_capped": "diag_mse_capped", "expr_distance_unbiased": "diag_ctrl_distance"}
LOW = {"nmae", "diag_mse_capped"}


def analyze():
    base = {M[k]: float(v) for k, v in pd.read_csv(f"{HERE}/data/bundle/baseline_agg.csv").set_index("statistic").loc["mean"].items() if k in M}
    rep = {M[k]: float(v) for k, v in pd.read_parquet(f"{HERE}/data/bundle/anchor_agg.parquet").set_index("metric")["replicate"].items() if k in M}
    rows, agg_rows = [], []
    present = [g for g in ALL_GROUPS if all(os.path.exists(f"{HERE}/scores/{g}_seed{s}/score.csv") for s in GEN_SEEDS)]
    for s in GEN_SEEDS:
        for g in present:
            d = f"{HERE}/scores/{g}_seed{s}"
            sc = pd.read_csv(f"{d}/score.csv").set_index("metric")["from_replicate"]
            raw = {M[k]: float(v) for k, v in pd.read_csv(f"{d}/agg_results.csv").set_index("statistic").loc["mean"].items() if k in M}
            agg_rows.append({"group": g, "seed": s, "avg_official_scale": float(sc["avg_score"]),
                             **{f"score_{M[k]}": float(sc[k]) for k in M if k in sc.index}, **{f"panel_raw_{k}": v for k, v in raw.items()},
                             "heuristic_fid_scaled_b05_NOT_OFFICIAL": (raw["fid"] - max(base["fid"], 0.5)) / (rep["fid"] - max(base["fid"], 0.5))})
            res = pd.read_csv(f"{d}/results.csv"); res = res[res.metric.isin(PER_TARGET)]
            for _, r in res.iterrows():
                m = PER_TARGET[r.metric]; u = float(r.value) if pd.notna(r.value) else np.nan
                rows.append({"group": g, "seed": s, "target": r.perturbation, "metric": m, "raw": u,
                             "scaled_with_panel_anchors_diagnostic": (u - base[m]) / (rep[m] - base[m]) if m in base and m in rep else np.nan})
    A = pd.DataFrame(agg_rows); T = pd.DataFrame(rows)
    A.to_csv(f"{HERE}/results_per_seed.csv", index=False); T.to_csv(f"{HERE}/results_per_target.csv", index=False)
    json.dump({"baseline_raw_panel": base, "replicate_raw_panel": rep}, open(f"{HERE}/anchors.json", "w"), indent=1)
    w = T.pivot_table(index=["seed", "target", "metric"], columns="group", values="raw").reset_index()
    pairs = [("full", "gm_plain", "shuffled")] + ([("strict", "strict_plain", "strict_shuffled")] if "strict_plain" in present else [])
    boot = []
    for tag, a_, b_ in pairs:
        w[f"diff_{tag}"] = w[a_] - w[b_]
    w.to_csv(f"{HERE}/paired_plain_vs_shuffled_per_seed.csv", index=False)
    for tag, a_, b_ in pairs:
        _boot(w, tag, boot)
    pd.DataFrame(boot).to_csv(f"{HERE}/bootstrap_plain_vs_shuffled.csv", index=False)
    log("analyze: wrote results_per_seed.csv, results_per_target.csv, paired_*.csv, bootstrap_plain_vs_shuffled.csv, anchors.json")


def _boot(w, tag, boot):
    """Seed-average each target's paired difference first, then bootstrap over the 12 targets."""
    m = w.groupby(["target", "metric"])[f"diff_{tag}"].agg(["mean", "std"]).reset_index()
    m.columns = ["target", "metric", "diff_seedmean", "diff_seed_sd"]
    m.to_csv(f"{HERE}/paired_{tag}_seedmean.csv", index=False)
    rng = np.random.default_rng(SEED)
    for metric, grp in m.groupby("metric"):
        v = grp["diff_seedmean"].dropna().to_numpy()
        if len(v) == 0: continue
        bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
        better = (v < 0) if metric in LOW else (v > 0)
        boot.append({"comparison": tag, "metric": metric, "n_targets": len(v), "mean_diff": v.mean(), "ci95_lo": np.quantile(bs, 0.025),
                     "ci95_hi": np.quantile(bs, 0.975), "targets_plain_better": int(better.sum()),
                     "lower_is_better": metric in LOW})


if __name__ == "__main__":
    {"prep": prep, "gen": lambda: gen(sys.argv[2:] or None), "strict_fit": strict_fit,
     "score": lambda: score(sys.argv[2:] or None), "analyze": analyze}[sys.argv[1]]()
