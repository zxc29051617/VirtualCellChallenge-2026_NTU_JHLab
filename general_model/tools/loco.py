"""Leave-one-cell-line-out evaluation with the official vcc2026 scorer (cell-eval2).

    python loco.py prep  --heldout HEK293T            # real set + baseline arm + real bundle
    python loco.py pred  --heldout HEK293T --method control_resample|transfer --sources HCT116
    python loco.py score --heldout HEK293T --method transfer
Line files: data/stage3/{xatlas_HCT116,xatlas_HEK293T,...}.h5ad (raw counts, VCC gene axis).
Like VCC, the controls given to a method are the same cells the scorer uses as reference.
"""
import argparse, json, os, subprocess, sys
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "/home/zxc29051617/VCC"
LINES = {"HCT116": "xatlas_HCT116", "HEK293T": "xatlas_HEK293T", "K562": "replogle_K562_gwps", "H1": "h1_H1",
         "KOLF": "kolf_KOLF", "KOLF200": "kolf_KOLF", "KOLFbig": "kolf_KOLF"}
CE2 = f"{ROOT}/evalenv/bin/cell-eval2"
N_CTRL, N_PRED, MIN_CELLS = 4000, 200, 50

p = argparse.ArgumentParser()
p.add_argument("cmd", choices=["prep", "pred", "score"])
p.add_argument("--heldout", required=True)
p.add_argument("--method", default="control_resample")
p.add_argument("--sources", nargs="*", default=[])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--roles", nargs="*", default=["vcc_target"], help="prep: which perturbation roles to evaluate")
p.add_argument("--effect-file", default="", help="pred effectfile: npz with targets, lfc (log2, VCC axis)")
p.add_argument("--scale", type=float, default=1.0, help="pred transfer: multiply transferred log2FC")
p.add_argument("--targets-file", default="", help="prep: restrict evaluated targets to this list (one per line)")
p.add_argument("--n-ctrl", type=int, default=0, help="prep: control cells in the real set (0 = 4000)")
p.add_argument("--n-pred", type=int, default=0, help="pred: predicted cells per target (0 = 200)")
p.add_argument("--max-cells", type=int, default=0, help="prep: cap perturbed cells per target (0 = all); "
               "the scorer needs ~12 GB at ~400M stored entries, too much for a 15 GB machine")
a = p.parse_args()
if a.n_ctrl: N_CTRL = a.n_ctrl
if a.n_pred: N_PRED = a.n_pred
D = f"{ROOT}/data/stage3/loco/{a.heldout}"; os.makedirs(D, exist_ok=True)


def load(line, backed=False):
    return ad.read_h5ad(f"{ROOT}/data/stage3/{LINES[line]}.h5ad", backed="r" if backed else None)


def pseudobulk_cpm(X):
    s = np.asarray(X.sum(0)).ravel()
    return s / s.sum() * 1e6


if a.cmd == "prep":
    rng = np.random.default_rng(0)
    A = load(a.heldout, backed=True)
    in_role = A.obs.role.isin(a.roles).values
    tg = A.obs[in_role].target_gene.value_counts()
    targets = sorted(tg.index[tg >= MIN_CELLS])
    if a.targets_file:
        keep = set(open(a.targets_file).read().split()); targets = [t for t in targets if t in keep]
    ctrl = np.flatnonzero(A.obs.is_control.values)
    ctrl = rng.choice(ctrl, min(N_CTRL, len(ctrl)), replace=False)
    pert = np.flatnonzero(A.obs.target_gene.isin(targets).values & in_role)
    if a.max_cells:
        r2 = np.random.default_rng(1)
        tg_of = A.obs.target_gene.values[pert]
        pert = np.concatenate([r2.permutation(pert[tg_of == t])[:a.max_cells] for t in targets])
    R = A[np.sort(np.concatenate([ctrl, pert]))].to_memory()
    R.obs = pd.DataFrame({"target_gene": np.where(R.obs.is_control, "non-targeting", R.obs.target_gene.astype(str))},
                         index=R.obs_names)
    R.var = pd.DataFrame(index=R.var_names)
    R.obs_names = [f"r{i}" for i in range(R.n_obs)]   # H1 barcodes repeat across the 2025 files
    R.write_h5ad(f"{D}/real.h5ad")
    json.dump({"targets": targets, "n_ctrl": int(len(ctrl)), "n_pert_cells": int(len(pert))}, open(f"{D}/real.json", "w"))
    print(f"{a.heldout}: {len(targets)} VCC targets with >={MIN_CELLS} cells, {len(ctrl)} controls, {len(pert)} perturbed cells")
    common = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting", "--input-type", "counts",
              "--set", "de.backend=pdex"]   # pin the DE engine: bundle and submission must match
    # prep-real-bundle wants the baseline ARM (its prediction h5ad), which `baseline` writes with --save-pred
    if not os.path.exists(f"{D}/baseline_pred.h5ad"):
      subprocess.run([CE2, "baseline", "-ar", f"{D}/real.h5ad", "-o", f"{D}/baseline",
                    "--save-pred", f"{D}/baseline_pred.h5ad", *common], check=True)
    subprocess.run([CE2, "prep-real-bundle", "--real", f"{D}/real.h5ad", "--baseline", f"{D}/baseline_pred.h5ad",
                    "-o", f"{D}/bundle", "--force", *common], check=True)

elif a.cmd == "pred":
    rng = np.random.default_rng(a.seed)
    R = ad.read_h5ad(f"{D}/real.h5ad")
    targets = json.load(open(f"{D}/real.json"))["targets"]
    C = sp.csr_matrix(R[R.obs.target_gene == "non-targeting"].X)
    lib = np.asarray(C.sum(1)).ravel()
    if a.method == "transfer":
        # log-fold-change of each target in the source lines (pseudobulk CPM), averaged over lines.
        # Source files are read backed, in row chunks, accumulating per-group count sums.
        lfc = {}
        for s in a.sources:
            S = load(s, backed=True)
            grp = np.where(S.obs.is_control.values, "__ctrl__", S.obs.target_gene.astype(str).values)
            keep = np.isin(grp, targets + ["__ctrl__"])
            names = sorted(set(grp[keep])); gi = {g: i for i, g in enumerate(names)}
            sums = np.zeros((len(names), S.n_vars)); ncell = np.zeros(len(names), int)
            for lo in range(0, S.n_obs, 20000):
                hi = min(S.n_obs, lo + 20000); k = np.flatnonzero(keep[lo:hi])
                if not len(k):
                    continue
                X = sp.csr_matrix(S.X[lo:hi][k]); g = np.array([gi[x] for x in grp[lo:hi][k]])
                ind = sp.csr_matrix((np.ones(len(k)), (g, np.arange(len(k)))), shape=(len(names), len(k)))
                sums += (ind @ X).toarray(); ncell += np.bincount(g, minlength=len(names))
            meas = S.var.measured.values
            cpm = sums / sums.sum(1, keepdims=True) * 1e6
            c = cpm[gi["__ctrl__"]]
            for t in targets:
                if t in gi and ncell[gi[t]] >= 10:
                    v = np.log2((cpm[gi[t]] + 1) / (c + 1)); v[~meas] = np.nan
                    lfc.setdefault(t, []).append(v)
            del S
        lfc = {t: a.scale * np.nan_to_num(np.nanmean(np.stack(v), 0)) for t, v in lfc.items()}
        print(f"transfer: effects for {len(lfc)}/{len(targets)} targets from {a.sources}")
    if a.method == "effectfile":
        z = np.load(a.effect_file, allow_pickle=True)
        # clip: a fold change learned from a gene that is near-silent in a source line's controls
        # (log2 CPM ~0.1) is not a reliable multiplier for a line that expresses it -- unclipped,
        # one such gene took a cell to 1.6e8 counts
        lfc = {t: np.clip(a.scale * np.nan_to_num(v), -4, 4) for t, v in zip(z["targets"], z["lfc"])}
        print(f"effectfile: effects for {sum(t in lfc for t in targets)}/{len(targets)} targets from {a.effect_file}")
        a.method = "effectfile_" + os.path.splitext(os.path.basename(a.effect_file))[0]
    blocks, labels = [], []
    for t in targets:
        # Match submission generation: sample without replacement when enough controls exist.
        draw = rng.choice(C.shape[0], N_PRED, replace=(N_PRED > C.shape[0]))
        sampled_lib = lib[draw]
        cells = C[draw].tocsr(copy=True)
        if (a.method == "transfer" or a.method.startswith("effectfile")) and t in lfc:
            fc = 2.0 ** lfc[t]
            f = fc[cells.indices]
            down = f < 1
            d = cells.data.astype(np.int64)
            d[down] = rng.binomial(d[down], f[down])                     # thin decreased genes
            cells.data = d.astype(np.float32)
            # add counts for increased genes: Poisson with mean (fc-1) * expected count in that cell
            up = np.flatnonzero(fc > 1)
            if len(up):
                # Use the original library, as make_effect_submission.py does, before thinning.
                rate = np.asarray(C.sum(0)).ravel() / lib.sum()
                mu = sampled_lib[:, None] * rate[None, up] * (fc[up] - 1)[None, :]
                add = sp.csr_matrix(rng.poisson(mu).astype(np.float32))          # columns = up genes
                cells = cells + sp.csr_matrix((add.data, up[add.indices], add.indptr), shape=cells.shape)
        cells.eliminate_zeros()
        blocks.append(cells); labels += [t] * N_PRED
    P = ad.AnnData(sp.vstack(blocks + [C]).tocsr(),
                   obs=pd.DataFrame({"target_gene": labels + ["non-targeting"] * C.shape[0]}),
                   var=pd.DataFrame(index=R.var_names))
    P.obs_names = [f"p{i}" for i in range(P.n_obs)]
    tag = a.method + (("_" + "+".join(a.sources)) if a.sources else "") + (f"_x{a.scale:g}" if a.scale != 1 else "")
    P.write_h5ad(f"{D}/pred_{tag}.h5ad")
    print(f"wrote pred_{tag}.h5ad: {P.n_obs} cells, {P.X.nnz/P.n_obs:.0f} nnz/cell")

elif a.cmd == "score":
    tag = a.method + (("_" + "+".join(a.sources)) if a.sources else "") + (f"_x{a.scale:g}" if a.scale != 1 else "")
    out = f"{D}/score_{tag}"
    common = ["--preset", "vcc2026", "--pert-col", "target_gene", "--control", "non-targeting", "--input-type", "counts",
              "--set", "de.backend=pdex"]   # pin the DE engine: bundle and submission must match
    subprocess.run([CE2, "run", "-ap", f"{D}/pred_{tag}.h5ad", "-ar", f"{D}/real.h5ad", "-o", out, *common], check=True)
    agg = [f for f in os.listdir(out) if f.endswith("agg.csv") or f.endswith("agg_results.csv")]
    print("agg files:", agg)
    subprocess.run([CE2, "score", "--user-agg", f"{out}/{agg[0]}", "--real-bundle", f"{D}/bundle", "-o", f"{out}/score.csv"], check=True)
    print(pd.read_csv(f"{out}/score.csv").to_string())
