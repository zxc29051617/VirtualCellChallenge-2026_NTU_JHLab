"""Per-line pseudobulk statistics, computed once from a line file (raw counts, VCC gene axis).

    python -m vcc.stats HCT116
writes data/stage3/stats/<line>.npz with, per perturbation group (targets + '__ctrl__'):
summed counts over cells, cell counts, and per-cell library sizes of the controls.
"""
import os, sys
import anndata as ad, numpy as np, scipy.sparse as sp

ROOT = "/home/zxc29051617/VCC/data/stage3"
FILES = {"HCT116": "xatlas_HCT116", "HEK293T": "xatlas_HEK293T", "K562": "replogle_K562_gwps", "H1": "h1_H1"}


def group_labels(obs):
    return np.where(obs.is_control.values, "__ctrl__", obs.target_gene.astype(str).values)


def compute(line, chunk=20000):
    A = ad.read_h5ad(f"{ROOT}/{FILES[line]}.h5ad", backed="r")
    grp = group_labels(A.obs)
    names = np.array(sorted(set(grp))); gi = {g: i for i, g in enumerate(names)}
    gidx = np.array([gi[g] for g in grp])
    sums = np.zeros((len(names), A.n_vars)); ncell = np.bincount(gidx, minlength=len(names))
    for lo in range(0, A.n_obs, chunk):
        hi = min(A.n_obs, lo + chunk)
        X = sp.csr_matrix(A.X[lo:hi]); g = gidx[lo:hi]
        ind = sp.csr_matrix((np.ones(hi - lo), (g, np.arange(hi - lo))), shape=(len(names), hi - lo))
        sums += (ind @ X).toarray()
    os.makedirs(f"{ROOT}/stats", exist_ok=True)
    np.savez_compressed(f"{ROOT}/stats/{line}.npz", names=names, sums=sums.astype(np.float32), ncell=ncell,
                        measured=A.var.measured.values, genes=np.array(A.var_names))
    print(f"{line}: {len(names)-1} perturbation groups + controls ({ncell[gi['__ctrl__']]} cells), {A.n_vars} genes")


def load(line):
    d = os.environ.get("STATS_DIR", f"{ROOT}/stats")       # overridable on a remote trainer
    z = np.load(f"{d}/{line}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def lfc_table(st, min_cells=30, pseudo=1.0):
    """log2((CPM_pert + 1) / (CPM_ctrl + 1)) per group; NaN for unmeasured genes."""
    cpm = st["sums"] / st["sums"].sum(1, keepdims=True) * 1e6
    names = list(st["names"]); c = cpm[names.index("__ctrl__")]
    keep = [i for i, n in enumerate(names) if n != "__ctrl__" and st["ncell"][i] >= min_cells]
    L = np.log2((cpm[keep] + pseudo) / (c + pseudo))
    L[:, ~st["measured"]] = np.nan
    return [names[i] for i in keep], L, c


if __name__ == "__main__":
    for line in sys.argv[1:]:
        compute(line)


def signal_variance(st, min_cells=30, pseudo=1.0):
    """Per-gene variance of perturbation effects across groups, minus the expected sampling
    noise of a pseudobulk log2 fold change (Poisson: var ~ (1/ln2)^2 (1/N_p + 1/N_c), N = summed counts)."""
    names, L, c = lfc_table(st, min_cells, pseudo)
    nm = list(st["names"]); keep = [nm.index(n) for n in names]
    Np = st["sums"][keep] + pseudo
    Nc = st["sums"][nm.index("__ctrl__")] + pseudo
    noise = (1 / np.log(2)) ** 2 * (1 / Np + 1 / Nc)
    Lc = L - np.nanmean(L, 0)                    # remove the line's generic response
    return np.nanmean(Lc ** 2, 0) - np.nanmean(noise, 0)
