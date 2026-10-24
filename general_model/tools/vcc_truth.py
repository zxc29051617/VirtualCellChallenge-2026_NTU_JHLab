"""A/B/C ground truth -> inputs for the general model and the local tests (prepared 2026-10-21, before the release).

    scDFM/.venv/bin/python -I tools/vcc_truth.py stats <truth.h5ad> <X> [--controls F] [--out F]
    scDFM/.venv/bin/python -I tools/vcc_truth.py loco  <truth.h5ad> <X> [--controls F] [--cells 200] [--n-ctrl 4000] [--out-dir D]

stats: per group summed counts over cells + cell counts in vcc/stats.py's format ('__ctrl__' = the non-targeting cells of
       the truth file when it has them, else the released controls data/vcc2026/context_<X>.h5ad), VCC gene axis, every
       gene measured. Default output data/stage3/stats/VCC_<X>.npz.
loco : a real set like data/stage3/loco/<line> (obs target_gene; up to --cells cells per target and --n-ctrl controls,
       seed 0; 0 = all) plus real.json with the target list, for bulk_eval / de_eval / the official scorer. Default output
       data/stage3/loco/VCC_<X>/.
The truth file is read in row chunks (backed mode). When it has a 'context' column with several values, only rows of
context <X> are used. Genes are reordered to the VCC axis when the file's order differs.
"""
import argparse, json, os, time
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "/home/zxc29051617/VCC"
PERT_COLS = ["target_gene", "target", "perturbation", "gene", "guide_target"]
CTRL_LABELS = {"non-targeting", "non_targeting", "NTC", "ntc", "control", "ctrl"}
CHUNK = 20000


def vcc_genes():
    return pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").iloc[:, 0].astype(str).tolist()


def open_h5ad(path, ctx):
    """backed AnnData, per-row labels ('non-targeting' for controls), rows of context ctx, gene permutation (or None)."""
    A = ad.read_h5ad(path, backed="r")
    col = next((c for c in PERT_COLS if c in A.obs.columns), None)
    if col is None:
        raise SystemExit(f"no perturbation column in {path}; obs columns: {list(A.obs.columns)}")
    lab = A.obs[col].astype(str).values
    lab = np.where(np.isin(lab, list(CTRL_LABELS)), "non-targeting", lab)
    rows = np.ones(A.n_obs, bool)
    if "context" in A.obs.columns and A.obs["context"].nunique() > 1:
        rows = A.obs["context"].astype(str).values == ctx
        if not rows.any():
            raise SystemExit(f"no rows of context {ctx} in {path}; contexts: {sorted(A.obs['context'].astype(str).unique())}")
    genes = vcc_genes(); vn = [str(g) for g in A.var_names]
    perm = None
    if vn != genes:
        pos = {g: i for i, g in enumerate(vn)}; miss = [g for g in genes if g not in pos]
        if miss:
            raise SystemExit(f"{len(miss)} VCC genes missing from {path}, e.g. {miss[:5]}")
        perm = np.array([pos[g] for g in genes])
    return A, lab, rows, perm


def read_rows(A, want, perm):
    """CSR float32 counts of the rows where want is True (in file order), read in contiguous chunks."""
    out = []
    for lo in range(0, A.n_obs, CHUNK):
        hi = min(A.n_obs, lo + CHUNK); m = want[lo:hi]
        if not m.any(): continue
        X = sp.csr_matrix(A.X[lo:hi])[np.flatnonzero(m)]
        out.append(X[:, perm] if perm is not None else X)
    return sp.vstack(out).tocsr().astype(np.float32) if out else None


def control_source(a, A, lab, rows, perm):
    """(AnnData, labels, row mask, perm) holding the controls: the truth's own NTC rows, else the released controls."""
    if ((lab == "non-targeting") & rows).any():
        return A, lab, rows & (lab == "non-targeting"), perm
    f = a.controls or f"{ROOT}/data/vcc2026/context_{a.context}.h5ad"
    Ac, labc, rowsc, permc = open_h5ad(f, a.context)
    print(f"truth has no non-targeting cells for {a.context}; controls from {f}", flush=True)
    return Ac, labc, rowsc & (labc == "non-targeting"), permc


def cmd_stats(a):
    A, lab, rows, perm = open_h5ad(a.truth, a.context); G = len(vcc_genes())
    names = sorted(set(lab[rows & (lab != "non-targeting")])); gi = {g: i for i, g in enumerate(names)}
    sums = np.zeros((len(names) + 1, G)); ncell = np.zeros(len(names) + 1, np.int64)     # row 0 = '__ctrl__'
    for lo in range(0, A.n_obs, CHUNK):
        hi = min(A.n_obs, lo + CHUNK); m = rows[lo:hi] & (lab[lo:hi] != "non-targeting")
        if not m.any(): continue
        X = sp.csr_matrix(A.X[lo:hi])[np.flatnonzero(m)]
        if perm is not None: X = X[:, perm]
        g = np.array([gi[t] for t in lab[lo:hi][m]]) + 1
        ind = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(len(names) + 1, len(g)))
        sums += (ind @ X).toarray(); ncell += np.bincount(g, minlength=len(names) + 1)
    Ac, labc, rowsc, permc = control_source(a, A, lab, rows, perm)
    Xc = read_rows(Ac, rowsc, permc); sums[0] = np.asarray(Xc.sum(0)).ravel(); ncell[0] = Xc.shape[0]
    out = a.out or f"{ROOT}/data/stage3/stats/VCC_{a.context}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, names=np.array(["__ctrl__"] + names), sums=sums.astype(np.float32), ncell=ncell,
                        measured=np.ones(G, bool), genes=np.array(vcc_genes()))
    nc = ncell[1:]
    print(f"wrote {out}: {len(names)} targets (cells per target median {np.median(nc):.0f}, min {nc.min()}, max {nc.max()}), "
          f"{ncell[0]} control cells", flush=True)


def cmd_loco(a):
    A, lab, rows, perm = open_h5ad(a.truth, a.context); rng = np.random.default_rng(0); genes = vcc_genes()
    targets = sorted(set(lab[rows & (lab != "non-targeting")]))
    keep = np.zeros(A.n_obs, bool); used = []
    for t in targets:
        idx = np.flatnonzero(rows & (lab == t))
        if len(idx) < a.min_cells: continue
        if a.cells and len(idx) > a.cells: idx = rng.choice(idx, a.cells, replace=False)
        keep[idx] = True; used.append(t)
    Xp = read_rows(A, keep, perm); lp = lab[keep]
    Ac, labc, rowsc, permc = control_source(a, A, lab, rows, perm)
    ci = np.flatnonzero(rowsc)
    if a.n_ctrl and len(ci) > a.n_ctrl: ci = np.sort(rng.choice(ci, a.n_ctrl, replace=False))
    wc = np.zeros(Ac.n_obs, bool); wc[ci] = True; Xc = read_rows(Ac, wc, permc)
    X = sp.vstack([Xc, Xp]).tocsr()
    obs = pd.DataFrame({"target_gene": np.concatenate([np.full(Xc.shape[0], "non-targeting"), lp])},
                       index=[f"r{i}" for i in range(X.shape[0])])
    R = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))
    R.uns["source"] = f"vcc2026 ground truth, context {a.context} ({os.path.basename(a.truth)})"
    out = a.out_dir or f"{ROOT}/data/stage3/loco/VCC_{a.context}"
    os.makedirs(out, exist_ok=True)
    R.write_h5ad(f"{out}/real.h5ad")
    json.dump({"targets": used, "n_ctrl": int(Xc.shape[0]), "n_pert_cells": int(Xp.shape[0])}, open(f"{out}/real.json", "w"))
    print(f"wrote {out}: {len(used)} targets ({len(targets) - len(used)} with < {a.min_cells} cells dropped), "
          f"{Xp.shape[0]} perturbed + {Xc.shape[0]} control cells, nnz {X.nnz:,}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["stats", "loco"])
    p.add_argument("truth"); p.add_argument("context")
    p.add_argument("--controls", default="", help="controls h5ad when the truth has no non-targeting cells")
    p.add_argument("--out", default="", help="stats: output npz")
    p.add_argument("--out-dir", default="", help="loco: output folder")
    p.add_argument("--cells", type=int, default=200, help="loco: max cells per target (0 = all)")
    p.add_argument("--n-ctrl", type=int, default=4000, help="loco: control cells (0 = all)")
    p.add_argument("--min-cells", type=int, default=50, help="loco: drop targets with fewer cells")
    a = p.parse_args(); t0 = time.time()
    {"stats": cmd_stats, "loco": cmd_loco}[a.cmd](a)
    print(f"done in {time.time() - t0:.0f}s")
