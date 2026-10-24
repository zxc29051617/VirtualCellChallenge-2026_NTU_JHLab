"""What would the DE members be if we chose the called genes differently? (2026-10-23)

    .venv/bin/python -I repro/model_v2_20261022/de_potential.py <variant> [subset]        (from scDFM/; run with evalenv/bin/python, which has cell_eval2)

Builds hypothetical prediction DE tables from the variant's own predictions (no new cells) and scores them with the
scorer's metric functions (de_sim.py): a ranking score per (target, gene) decides the order (reach) and the top n per
target are called significant (fid, jaccard), with the predicted sign. log2 fold changes stay those of the real run, with
the sign of the called genes set to the predicted sign, so nmae is only comparable within a row group.
"""
import os, sys
import numpy as np, pandas as pd, polars as pl, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import de_sim as DS

EFF = "/home/zxc29051617/VCC/data/stage3/effects"
v = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ref = DS.Ref(v, k)
P0 = ref.pred.to_pandas(); R0 = ref.real.to_pandas()
targets = sorted(P0.target.unique()); feats = sorted(P0.feature.unique())
gi = {g: i for i, g in enumerate(pd.read_csv('/home/zxc29051617/VCC/data/vcc2026/gene_names.csv').gene_name.astype(str))}; cols = np.array([gi.get(f, -1) for f in feats])
fi = {f: j for j, f in enumerate(feats)}; tix = {t: i for i, t in enumerate(targets)}

def mat(df, col):
    M = np.full((len(targets), len(feats)), np.nan)
    M[df.target.map(tix).values, df.feature.map(fi).values] = df[col].values
    return M

Lp, Pp = mat(P0, "log2_fold_change"), mat(P0, "p_value")
Lr, Sr = mat(R0, "log2_fold_change"), mat(R0, "p_adj")
own = np.array([[f == t for f in feats] for t in targets])
Rsig = (Sr < 0.05) & ~own; n_conf = Rsig.sum(1)
Ppadj = mat(P0, "p_adj"); n_cur = ((Ppadj < 0.05) & ~own).sum(1)


def load(name):
    z = np.load(f"{EFF}/mock_{v}_s{k}_{name}.npz", allow_pickle=True); tl = [str(t) for t in z["targets"]]
    X = z["lfc"].astype(np.float64)[[tl.index(t) for t in targets]]
    out = np.zeros((len(targets), len(feats))); ok = cols >= 0; out[:, ok] = X[:, cols[ok]]; return out

T = load("bulk_g1_H"); C = load("cells_agree"); O = load("oc")
Xc = sp.csr_matrix(ad.read_h5ad(f"/home/zxc29051617/VCC/data/mock_h1/s{k}/context_H.h5ad").X)
lib = np.asarray(Xc.sum(1)).ravel(); cpm = sp.diags(1e6 / lib) @ Xc
mu_all = np.asarray(cpm.mean(0)).ravel(); expr = np.zeros(len(feats)); expr[cols >= 0] = mu_all[cols[cols >= 0]]
frac_nz = np.zeros(len(feats)); frac_nz[cols >= 0] = np.asarray((Xc > 0).mean(0)).ravel()[cols[cols >= 0]]


def table(score, sign, n_call, lfc=None):
    """score: higher = more confident; sign: +-1 per entry; n_call: per target number of called genes."""
    rows = []
    L = np.where(np.isfinite(Lp), Lp, 0.0) if lfc is None else lfc
    for i, t in enumerate(targets):
        s = np.where(own[i], -np.inf, score[i]); order = np.argsort(-s); rank = np.empty(len(feats)); rank[order] = np.arange(len(feats))
        called = rank < n_call[i]
        p = np.where(called, 1e-12 * (rank + 1), 0.06 + 1e-6 * rank)
        l = np.where(called, np.abs(L[i]) * sign[i], L[i])
        l = np.where(called & (l == 0), sign[i] * 1e-3, l)
        rows.append(pd.DataFrame({"target": t, "feature": feats, "log2_fold_change": l, "p_value": p, "p_adj": p}))
    return pd.concat(rows, ignore_index=True)


def show(name, tab):
    r, sc = DS.score(ref, tab)
    print(f"{name:42s} " + "  ".join(f"{m} {r[m]:.4f} ({sc[m]:+.3f})" for m in ("fid", "reach", "jac", "nmae")), flush=True)


print(f"subset {k}: reference calls per target median {np.median(n_conf):.0f}, ours median {np.median(n_cur):.0f}")
show("actual run (scorer DE table)", P0)
cur_score = -np.log10(np.clip(Pp, 1e-300, 1)); sgn_cur = np.sign(np.where(np.isfinite(Lp), Lp, 0))
show("actual ranking, actual count", table(cur_score, sgn_cur, n_cur))
show("actual ranking, count = N_conf (oracle)", table(cur_score, sgn_cur, n_conf))
for nm, S, sg in (("|cells| x sqrt(expr)", np.abs(C) * np.sqrt(expr), np.sign(C)), ("|totals| x sqrt(expr)", np.abs(T) * np.sqrt(expr), np.sign(T)),
                  ("|cells|", np.abs(C), np.sign(C)), ("|totals|", np.abs(T), np.sign(T)),
                  ("|cells| x sqrt(frac nonzero)", np.abs(C) * np.sqrt(frac_nz), np.sign(C))):
    sg = np.where(sg == 0, 1.0, sg)
    show(f"{nm}, actual count", table(S, sg, n_cur))
    show(f"{nm}, count = N_conf (oracle)", table(S, sg, n_conf))
# oracle signs on our ranking: how much is lost to wrong signs
show("actual ranking, TRUE signs, actual count", table(cur_score, np.where(np.isfinite(Lr), np.sign(Lr), 1.0), n_cur))
