"""Fast proxy of the scorer's mse member from gene totals alone (2026-10-23), to explore totals variants without
building cells. The dual render sets every gene's total over a target's 400 cells exactly to round(400 x ref_mean x
2^lfc_bulk), so the prediction's pseudobulk is known from the totals file.

    mu = log1p(5e4 x sums / sum of sums)                              (the scorer's bulk_lognorm comparator)
    E_p = ||mu_pred,p - mu_real,p||^2 - C_real,p - r x min(C_pred, C_real,p)    (C: delete-1 jackknife traces)
    r = min(1, B_pred / sum_p min(C_pred, C_real,p)), B_pred = across-target centred sum of squares of mu_pred (#348)
    raw = sum_p E_p / sum_p E_base,p,  base = the context mean of the real perturbed cells (the scorer's baseline; its
    spread is 0, so it gets no prediction-side correction)
own target gene excluded. Validated against the official raw values of finished mock runs (see __main__).

    from mse_proxy import Proxy; P = Proxy(k); P.raw(lfc_bulk)   # lfc_bulk: targets x genes (log2), P.targets order
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

MOCK = "/home/zxc29051617/VCC/data/mock_h1"; EFF = "/home/zxc29051617/VCC/data/stage3/effects"; TS = 5e4


def _mu(S):
    tot = S.sum(-1, keepdims=True); return np.log1p(TS * S / np.maximum(tot, 1e-12))


def _jk(Y):
    """Delete-1 jackknife trace of the pseudobulk mu of the cells Y (as tools/make_bulk_effects.jk_trace)."""
    s = np.asarray(Y.sum(0)).ravel(); T = s.sum(); f = s / T; L = np.asarray(Y.sum(1)).ravel()
    x2 = np.asarray(Y.multiply(Y).sum(0)).ravel(); xl = np.asarray(Y.T @ L).ravel()
    varf = np.maximum(x2 - 2 * f * xl + f ** 2 * (L ** 2).sum(), 0) / T ** 2
    return float(((TS ** 2) * varf / (1 + TS * f) ** 2).sum())


class Proxy:
    def __init__(self, k):
        d = f"{MOCK}/s{k}"
        R = ad.read_h5ad(f"{d}/real.h5ad"); X = sp.csr_matrix(R.X)
        lab = R.obs["target_gene"].astype(str).values; self.genes = list(R.var_names); gi = {g: i for i, g in enumerate(self.genes)}
        self.targets = sorted(set(lab) - {"non-targeting"})
        ind = sp.csr_matrix((np.ones(len(lab)), (pd.Categorical(lab, categories=self.targets + ["non-targeting"]).codes, np.arange(len(lab)))),
                            shape=(len(self.targets) + 1, len(lab)))
        S = np.asarray((ind @ X).todense()); self.n = np.asarray(ind.sum(1)).ravel()
        self.Sr = S[:-1]; ctrl = S[-1]; self.mu_c = _mu(ctrl[None])[0]
        C = sp.csr_matrix(ad.read_h5ad(f"{d}/context_H.h5ad").X); self.ref_mean = np.asarray(C.sum(0)).ravel() / C.shape[0]
        self.own = np.array([gi.get(t, -1) for t in self.targets])
        self.mu_r = _mu(self.Sr)
        self.c_r = np.array([_jk(X[np.flatnonzero(lab == t)]) for t in self.targets])
        rng = np.random.default_rng(0); self.c_p = float(np.mean([_jk(C[rng.choice(C.shape[0], 400, replace=False)]) for _ in range(5)]))
        base = self.Sr.sum(0)                                            # pooled perturbed cells (context mean)
        self.mu_b = _mu(base[None])[0]
        self.E_base = self._err(np.repeat(self.mu_b[None], len(self.targets), 0))

    def _err(self, mu_p):
        D = (mu_p - self.mu_r) ** 2
        for i, g in enumerate(self.own):
            if g >= 0: D[i, g] = 0.0
        cp = np.minimum(self.c_p, self.c_r); B = ((mu_p - mu_p.mean(0, keepdims=True)) ** 2).sum()
        r = min(1.0, B / cp.sum())
        return D.sum(1) - self.c_r - r * cp

    def raw(self, lfc, n=400):
        T = n * self.ref_mean[None] * 2.0 ** np.clip(lfc, -4, 4)
        return float(self._err(_mu(T)).sum() / self.E_base.sum())

    def pds(self, lfc, n=400):
        """The scorer's pds_cosine (bulk_lognorm deltas against the real controls, every panel target gene removed from the
        features, cosine distance, midrank ties, rank / (n - 1)), from the totals alone."""
        T = n * self.ref_mean[None] * 2.0 ** np.clip(lfc, -4, 4); mp = _mu(T)
        keep = np.ones(mp.shape[1], bool); keep[self.own[self.own >= 0]] = False
        Dp = (mp - self.mu_c[None])[:, keep]; Dr = (self.mu_r - self.mu_c[None])[:, keep]
        Dp /= np.maximum(np.linalg.norm(Dp, axis=1, keepdims=True), 1e-12); Dr /= np.maximum(np.linalg.norm(Dr, axis=1, keepdims=True), 1e-12)
        dist = 1 - Dp @ Dr.T; own = np.diag(dist)[:, None]
        rank = (dist < own).sum(1) + 0.5 * ((dist == own).sum(1) - 1)
        return float(np.mean(1 - rank / max(len(self.targets) - 1, 1)))

    def load(self, path):
        z = np.load(path, allow_pickle=True); tl = [str(t) for t in z["targets"]]
        return z["lfc"].astype(np.float64)[[tl.index(t) for t in self.targets]]


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    P = Proxy(k)
    import json
    for v in sys.argv[2:]:
        off = json.load(open(f"/home/zxc29051617/VCC/pertrelay/model/mock/mock_{v}_s{k}.json"))["raw"]["expr_mse_unbiased_capped_norm"]
        r = json.load(open(f"/home/zxc29051617/VCC/pertrelay/model/mock/mock_{v}_s{k}.json"))["raw"]
        L = P.load(f'{EFF}/mock_{v}_s{k}_bulk_g1_H.npz')
        print(f"{v:22s} mse proxy {P.raw(L):.4f} official {off:.4f} | pds proxy {P.pds(L):.4f} official {r['pds_cosine']:.4f}")
