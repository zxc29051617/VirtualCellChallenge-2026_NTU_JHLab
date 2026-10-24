"""How much of a knockdown's response is a shift in cell-state composition? (2026-10-23)

    .venv/bin/python -I repro/model_v2_20261022/state_shift.py [subset]        (from scDFM/)

On a mock H1 subset (real perturbed cells + 4,000 controls): score every cell for cell-cycle S and G2M programs (Tirosh
2015 gene lists, mean log1p CP10k minus the cell's mean over all genes), then for each target reweight the CONTROL cells
(exponential tilting) so their mean S / G2M scores match the target's real cells, and compare the reweighted controls'
pseudobulk change with the target's real change (log2 CP10k means, genes expressed in controls). If the reweighting
explains a large share, a cell-type-specific renderer can carry that part: shift the new cell type's own control
population instead of transferring gene-level changes.
"""
import sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
S_GENES = "MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8".split()
G2M_GENES = "HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA".split()
R = ad.read_h5ad(f"/home/zxc29051617/VCC/data/mock_h1/s{k}/real.h5ad")
X = sp.csr_matrix(R.X, dtype=np.float32); del R.X                                   # one copy only (15 GB box)
lib = np.asarray(X.sum(1)).ravel().astype(np.float64); sc = 1e4 / lib
genes = list(R.var_names); gi = {g: i for i, g in enumerate(genes)}; G = len(genes)
rowsc = np.repeat(sc, np.diff(X.indptr)).astype(np.float32)
logv = np.log1p(X.data * rowsc)                                                      # log1p CP10k of the stored entries
cell_mean = np.add.reduceat(np.r_[logv, 0], X.indptr[:-1]) * (np.diff(X.indptr) > 0) / G
def score(gl):
    idx = np.array([gi[g] for g in gl if g in gi]); sub = X[:, idx].tocsr(); r = np.repeat(sc[:, None], 1, 1)
    sub = sp.csr_matrix((np.log1p(sub.data * np.repeat(sc, np.diff(sub.indptr))), sub.indices, sub.indptr), shape=sub.shape)
    return np.asarray(sub.mean(1)).ravel() - cell_mean
Z = np.stack([score(S_GENES), score(G2M_GENES)], 1)
lab = R.obs["target_gene"].astype(str).values; ctrl = lab == "non-targeting"; ci = np.flatnonzero(ctrl)
Zc = Z[ctrl]
def cp10k_mean(rows, w=None):
    v = np.zeros(X.shape[0]); v[rows] = sc[rows] * (np.full(len(rows), 1 / len(rows)) if w is None else w)
    return np.asarray(X.T @ v).ravel()
mu_c = cp10k_mean(ci); expressed = mu_c > 0.05          # CP10k
lc = np.log2(mu_c + 1e-3)
def tilt(target_mean, iters=200):
    th = np.zeros(2)
    for _ in range(iters):
        w = np.exp((Zc - Zc.mean(0)) @ th); w /= w.sum()
        m = w @ Zc; cov = (Zc - m).T @ ((Zc - m) * w[:, None]) + 1e-6 * np.eye(2)
        th += np.linalg.solve(cov, target_mean - m) * 0.5
    return w
rows = []
for t in sorted(set(lab[~ctrl])):
    m = lab == t
    zt = Z[m].mean(0); w = tilt(zt)
    real = np.log2(cp10k_mean(np.flatnonzero(m)) + 1e-3) - lc
    comp = np.log2(cp10k_mean(ci, w) + 1e-3) - lc
    e = expressed
    big = e & (np.abs(real) > 0.3)
    rows.append(dict(target=t, n=int(m.sum()), dS=zt[0] - Zc[:, 0].mean(), dG2M=zt[1] - Zc[:, 1].mean(),
                     corr=np.corrcoef(real[e], comp[e])[0, 1], corr_big=np.corrcoef(real[big], comp[big])[0, 1] if big.sum() > 20 else np.nan,
                     r2=1 - np.sum((real[e] - comp[e]) ** 2) / np.sum(real[e] ** 2), ess=1 / np.sum(w ** 2)))
T = pd.DataFrame(rows).set_index("target")
pd.set_option("display.width", 200)
print(T.describe().round(3).to_string())
print(T.sort_values("corr", ascending=False).head(12).round(3).to_string())
