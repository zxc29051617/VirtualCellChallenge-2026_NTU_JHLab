"""Fixed samples of single-cell control (non-targeting) cells per context, on the 18,533-gene challenge axis.

    python sample_controls.py            (scDFM venv: anndata, scipy)

Per context: N_CELLS=2000 control cells drawn without replacement with seed [20261013, context index].
Held-out contexts (H1, HCT116, KOLF) draw from the NTC cells of the local hold-out real sets (the controls that a
method would be given for a new context); other contexts draw from their control cells in the stage-3 / raw files.
VIPK562 controls are gathered from the RDS (CSC) by tools/rds_stream. Contexts with only pseudobulk data
(CD4_*, FengNF, FengF) have no single-cell controls here and are listed in controls/availability.json.
Output: controls/<ctx>.npz (csr 'counts', 'measured' bool over the gene axis, 'cells').
"""
import json, os, pickle, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "/home/zxc29051617/VCC"; HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/controls"; os.makedirs(OUT, exist_ok=True)
N_CELLS, SEED = 2000, 20261013
GENES = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); GI = {g: i for i, g in enumerate(GENES)}
ORDER = ["H1", "HCT116", "KOLF", "HEK293T", "K562", "K562ess", "HepG2", "Jurkat", "VIPK562"]
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)


def save(ctx, X, measured, src):
    X = sp.csr_matrix(X, dtype=np.float32); X.eliminate_zeros()
    sp.save_npz(f"{OUT}/{ctx}_counts.npz", X)
    np.save(f"{OUT}/{ctx}_measured.npy", measured)
    log(f"{ctx}: {X.shape[0]} cells, mean UMI {np.asarray(X.sum(1)).mean():.0f}, nnz/cell {X.nnz / X.shape[0]:.0f}, "
        f"{int(measured.sum())} measured genes ({src})")
    return {"cells": int(X.shape[0]), "measured_genes": int(measured.sum()), "source": src}


def pick(n, ctx):
    return np.sort(np.random.default_rng([SEED, ORDER.index(ctx)]).choice(n, min(N_CELLS, n), replace=False))


def from_real(ctx, loco):
    A = ad.read_h5ad(f"{ROOT}/data/stage3/loco/{loco}/real.h5ad")
    idx = np.flatnonzero(A.obs.target_gene.astype(str).values == "non-targeting"); sel = idx[pick(len(idx), ctx)]
    X = sp.csr_matrix(A.X[sel]); meas = np.asarray(X.sum(0)).ravel() > 0
    return X, meas, f"data/stage3/loco/{loco}/real.h5ad non-targeting"


def from_stage3(ctx, f):
    A = ad.read_h5ad(f"{ROOT}/data/stage3/{f}.h5ad", backed="r")
    idx = np.flatnonzero(A.obs.is_control.values); sel = idx[pick(len(idx), ctx)]
    X = sp.csr_matrix(A.X[sel]); meas = A.var.measured.values.astype(bool) if "measured" in A.var else np.asarray(X.sum(0)).ravel() > 0
    return X, meas, f"data/stage3/{f}.h5ad is_control"


def from_raw(ctx, path):
    A = ad.read_h5ad(f"{ROOT}/data/external/{path}", backed="r")
    idx = np.flatnonzero(A.obs.gene.astype(str).values == "non-targeting"); sel = idx[pick(len(idx), ctx)]
    X = sp.csr_matrix(A.X[sel]); col = np.array([GI.get(str(g), -1) for g in A.var.gene_name.astype(str)])
    ok = np.flatnonzero(col >= 0)
    M = sp.csr_matrix((np.ones(len(ok)), (np.arange(len(ok)), col[ok])), shape=(len(ok), len(GENES)))
    Y = (X[:, ok] @ M).tocsr(); meas = np.zeros(len(GENES), bool); meas[col[ok]] = True
    return Y, meas, f"data/external/{path} non-targeting (symbols matched)"


def vip_controls():
    sys.path.insert(0, f"{ROOT}/tools"); import rds_stream as R
    o = pickle.load(open("/home/zxc29051617/VCC/data/external/viperturb/vip_skel.pkl", "rb"))
    md = o.attr["meta.data"]; cols = dict(zip(md.attr["names"].value, md.value)); gene = np.array(cols["gene"].value)
    rna = o.attr["assays"].value[0]; cnt = rna.attr["layers"].value[0]
    p = np.asarray(cnt.attr["p"].value, np.int64); feat = np.array(rna.attr["features"].attr["dimnames"].value[0].value)
    idx = np.flatnonzero(gene == "NO-TARGET"); sel = idx[pick(len(idx), "VIPK562")]
    lens = p[sel + 1] - p[sel]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    want = np.zeros(int(p[-1]) + 1, bool)
    for s, l, c in zip(starts, lens, sel): want[p[c]:p[c] + l] = True
    state = {"pos": 0, "rows": [], "vals": [], "xpos": 0}
    def cb(k, typ, arr):
        n = len(arr)
        if k == 0:
            m = want[state["pos"]:state["pos"] + n]; state["rows"].append(arr[m].astype(np.int64)); state["pos"] += n
        elif k == 1:
            m = want[state["xpos"]:state["xpos"] + n]; state["vals"].append(arr[m].astype(np.float32)); state["xpos"] += n
            if state["xpos"] >= int(p[-1]): raise StopIteration
    try: R.stream(f"{ROOT}/data/external/viperturb/genome_wide_filtered.rds", cb)
    except StopIteration: pass
    rows = np.concatenate(state["rows"]); vals = np.concatenate(state["vals"])
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    X = sp.csr_matrix((vals, rows, indptr), shape=(len(sel), len(feat)))
    col = np.array([GI.get(str(g), -1) for g in feat]); ok = np.flatnonzero(col >= 0)
    M = sp.csr_matrix((np.ones(len(ok)), (np.arange(len(ok)), col[ok])), shape=(len(ok), len(GENES)))
    Y = (X[:, ok] @ M).tocsr(); meas = np.zeros(len(GENES), bool); meas[col[ok]] = True
    return Y, meas, "data/external/viperturb/genome_wide_filtered.rds NO-TARGET (symbols matched)"


JOBS = {"H1": lambda: from_real("H1", "H1"), "HCT116": lambda: from_real("HCT116", "HCT116"),
        "KOLF": lambda: from_real("KOLF", "KOLF200"), "HEK293T": lambda: from_stage3("HEK293T", "xatlas_HEK293T"),
        "K562": lambda: from_stage3("K562", "replogle_K562_gwps"),
        "K562ess": lambda: from_raw("K562ess", "replogle2022/K562_essential_raw_singlecell_01.h5ad"),
        "HepG2": lambda: from_raw("HepG2", "nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad"),
        "Jurkat": lambda: from_raw("Jurkat", "nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad"), "VIPK562": vip_controls}

if __name__ == "__main__":
    avail = {}
    for ctx in ORDER:
        if os.path.exists(f"{OUT}/{ctx}_counts.npz"):
            log(f"{ctx}: exists, skipped"); continue
        X, meas, src = JOBS[ctx](); avail[ctx] = save(ctx, X, meas, src)
    prev = json.load(open(f"{OUT}/availability.json")) if os.path.exists(f"{OUT}/availability.json") else {}
    prev.update(avail)
    for c in ["CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr", "FengNF", "FengF"]:
        prev[c] = {"cells": 0, "source": "pseudobulk only - no single-cell controls; attribution features missing (neutral)"}
    json.dump(prev, open(f"{OUT}/availability.json", "w"), indent=1)
