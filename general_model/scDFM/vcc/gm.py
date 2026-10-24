"""General response model v1 for an unseen cell line, from its control cells only.

For target p in a context with control profile x (CP50k per gene), the predicted log fold change is
    ln r[g] = OWN                                   g = p                (CRISPRi knocks the target down)
            + OWN * beta(d(g, p), bidirectional)     g within 50 kb of p  (cis: KRAB silences neighbours)
            + ALPHA * s_p[g]                         all other genes      (trans: learned signature of p)
Learned from the training contexts:
  beta : per TSS-distance bin (and head-to-head partners), least squares of neighbour ln r on own ln r;
  s_p  : ridge regression over contexts: argmin_s sum_c w_c ||M_cp (k_cp y_cp - s)||^2 + LAM ||s||^2, where y_cp is
         context c's ln r with its common mode (mean over perturbations) removed, k_cp rescales c's knockdown of p
         to OWN, and M_cp masks unmeasured / unexpressed genes and the own + cis genes.
At prediction time the model sees only the target name and the new context's controls.

    python -m vcc.gm loco                       # leave-one-cell-type-out on every context group
(historical protocol: here HCT116 / HEK293T and FengNF / FengF / KOLF are separate groups, so a same-screen sibling can
still be a source; the strict 7-fold benchmark with sibling groups hidden together is repro/model_v2_20261022/bench.py)
"""
import os, sys
import numpy as np
from vcc.contexts import load_all
from vcc.genome import on_axis

DATA = os.environ.get("VCC_DATA_DIR", "/home/zxc29051617/VCC/data/vcc2026")   # challenge controls + target list


def challenge_targets(data=None):
    import pandas as pd
    return list(dict.fromkeys(pd.read_csv(f"{data or DATA}/pert_counts.csv")["target_gene"].astype(str)))


EPS = 0.5                     # CP50k pseudocount for log ratios
X_MIN = 0.25                  # CP50k (= 5 CPM): genes below this in a context carry no usable ratio
CIS_BINS = np.array([0, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4])
GROUPS = [["K562", "K562ess", "VIPK562"], ["HCT116"], ["HEK293T"], ["H1"], ["CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr"],
          ["HepG2"], ["Jurkat"], ["FengNF", "FengF"], ["KOLF"], ["TeloHAEC"]]
# weighting by cell TYPE: all iPSC screens (Feng donors, KOLF2.1J) share one unit of weight
CELLTYPES = [["K562", "K562ess", "VIPK562"], ["HCT116"], ["HEK293T"], ["H1"], ["CD4_Rest", "CD4_Stim8hr", "CD4_Stim48hr"],
             ["HepG2"], ["Jurkat"], ["FengNF", "FengF", "KOLF"], ["TeloHAEC"]]


_RCACHE = {}


def ratios(c):
    """ln r per perturbation x gene (NaN where unmeasured or unexpressed), control CP50k. Cached per context."""
    if c.name in _RCACHE: return _RCACHE[c.name]
    _RCACHE[c.name] = _ratios(c); return _RCACHE[c.name]


def _ratios(c):
    xc = np.expm1(c.ctrl.astype(np.float64)); xp = np.expm1(c.ctrl[None] + c.delta.astype(np.float64))
    L = np.log((xp + EPS) / (xc + EPS))
    L[:, ~(c.measured & (xc >= X_MIN))] = np.nan
    return L.astype(np.float32), xc


class Genome:
    def __init__(self, genes):
        self.genes = np.asarray(genes); self.idx = {g: i for i, g in enumerate(self.genes)}
        self.chr, self.tss, self.strand = on_axis(self.genes)

    def cis(self, p):
        """(gene indices, bin index) of genes near p's TSS; bin len(CIS_BINS)-1 = head-to-head partner (<2 kb)."""
        if not hasattr(self, "_cis"): self._cis = {}
        if p not in self._cis: self._cis[p] = self._cis_uncached(p)
        return self._cis[p]

    def _cis_uncached(self, p):
        k = self.idx.get(p)
        if k is None or self.tss[k] < 0: return np.array([], int), np.array([], int)
        m = (self.chr == self.chr[k]) & (self.tss >= 0); m[k] = False
        d = np.abs(self.tss - self.tss[k]); m &= d < CIS_BINS[-1]
        g = np.flatnonzero(m); b = np.searchsorted(CIS_BINS, d[g], side="right") - 1
        h2h = (d[g] < 2e3) & (self.strand[g] == -self.strand[k]); b[h2h] = len(CIS_BINS) - 1
        return g, b


_GENOME = {}


class GeneralModel:
    def __init__(self, own=-1.9, alpha=1.0, lam=1.0, use_cis=True, use_trans=True, ctx_weight=None,
                 per_celltype=True, kcap=3.0, sim_tau=0.0, zgate=0.0, generic=0.0, n0=0.0, weight_groups="loco",
                 shuffle_trans=False, topk=0, alpha_rest=None):
        self.own, self.alpha, self.lam, self.use_cis, self.use_trans = own, alpha, lam, use_cis, use_trans
        self.sim_tau = sim_tau          # > 0: weight training contexts by exp(tau * corr(controls)) to the new context
        self.zgate = zgate              # > 0: keep only trans genes whose signature is >= zgate standard errors from 0
        self.generic = generic          # > 0: add this x the training contexts' mean response to the VCC targets
        self.n0 = n0                    # > 0: weight each (context, target) by ncell / (ncell + n0) (shallow screens count less)
        self.shuffle_trans = shuffle_trans  # negative control: each target gets ANOTHER target's trans signature
        self.topk, self.alpha_rest = topk, alpha_rest   # topk > 0: only the k largest |s_p| genes keep ALPHA, the rest ALPHA_REST
        self.ctx_weight = dict(ctx_weight or {}); self.kcap = kcap
        if per_celltype:          # contexts of one cell type share one unit of weight (CD4 states, K562 screens)
            for grp in (CELLTYPES if weight_groups == "celltype" else GROUPS):
                for g in grp: self.ctx_weight.setdefault(g, 1.0 / len(grp))

    def fit(self, C, targets=None):
        """targets: only these perturbations get a trans signature (all, if None); cis uses every perturbation."""
        tset = None if targets is None else set(targets)
        any_c = next(iter(C.values()))
        self.G = _GENOME.setdefault("g", Genome(any_c.genes)); nb = len(CIS_BINS)
        num = np.zeros(nb); den = np.zeros(nb)
        acc_s, acc_w, acc_v = {}, {}, {}
        self.parts, self.ctrl_profiles = {}, {n: (c.ctrl, c.measured) for n, c in C.items()}
        vcc = set(challenge_targets())
        gnum = np.zeros(len(self.G.genes)); gden = np.zeros(len(self.G.genes))
        for name, c in C.items():
            L, xc = ratios(c)
            cm = np.nanmean(L, 0); Y = L - cm[None]
            iv = np.flatnonzero(np.isin(c.perts, list(vcc)))
            if len(iv) >= 30:                       # generic response = mean over the VCC targets in this context
                gv = np.nanmean(L[iv], 0); okg = np.isfinite(gv); wg = self.ctx_weight.get(name, 1.0)
                gnum[okg] += wg * gv[okg]; gden[okg] += wg
            gi = self.G.idx; wc = self.ctx_weight.get(name, 1.0)
            for j, p in enumerate(c.perts):
                k = gi.get(p); own_obs = L[j, k] if k is not None else np.nan
                g, b = self.G.cis(p)
                if np.isfinite(own_obs) and own_obs < -0.5 and len(g):
                    y = Y[j, g]; ok = np.isfinite(y)
                    np.add.at(num, b[ok], y[ok] * own_obs); np.add.at(den, b[ok], own_obs ** 2)
                wcp = wc * (c.ncell[j] / (c.ncell[j] + self.n0) if self.n0 > 0 else 1.0)
                kscale = np.clip(self.own / own_obs, 0.5, self.kcap) if np.isfinite(own_obs) and own_obs < -0.3 else 1.0
                if tset is not None and p not in tset: continue
                y = Y[j].copy(); y[g] = np.nan
                if k is not None: y[k] = np.nan
                ok = np.isfinite(y)
                if p not in acc_s:
                    acc_s[p] = np.zeros(len(y), np.float32); acc_w[p] = np.zeros(len(y), np.float32)
                    if self.zgate > 0: acc_v[p] = np.zeros(len(y), np.float32)
                acc_s[p][ok] += wcp * kscale * y[ok]; acc_w[p][ok] += wcp
                if self.zgate > 0: acc_v[p][ok] += (wcp * kscale) ** 2 * np.nan_to_num(c.var[j][ok], nan=1.0)
                if self.sim_tau > 0:
                    self.parts.setdefault(p, []).append((name, wcp, np.where(ok, kscale * y, 0).astype(np.float32), ok))
        self.beta = num / np.maximum(den, 1e-9)
        self.gen = (gnum / np.maximum(gden, 1e-9)).astype(np.float32)
        self.sig = {}
        for p in acc_s:
            sgn = acc_s[p] / (acc_w[p] + self.lam)
            if self.zgate > 0:
                se = np.sqrt(acc_v[p]) / (acc_w[p] + self.lam)
                sgn = np.where(np.abs(sgn) >= self.zgate * np.maximum(se, 1e-6), sgn, 0.0)
            self.sig[p] = sgn.astype(np.float32)
        return self

    def context_weights(self, ctrl):
        """exp(tau * Pearson r) between the new context's control profile and each training context's (expressed genes)."""
        w = {}
        for n, (e, meas) in self.ctrl_profiles.items():
            m = meas & ((e > 0.5) | (ctrl > 0.5))
            w[n] = float(np.exp(self.sim_tau * (np.corrcoef(e[m], ctrl[m])[0, 1] - 1)))
        return w

    def predict(self, targets, ctrl=None):
        """ln r (targets x genes). ctrl: the new context's control expression (scorer space), used if sim_tau > 0."""
        out = np.zeros((len(targets), len(self.G.genes)), np.float32)
        cw = self.context_weights(ctrl) if (self.sim_tau > 0 and ctrl is not None) else None
        donor = list(targets)
        if self.shuffle_trans:                     # a derangement: nobody keeps its own signature
            rng = np.random.default_rng(0); perm = rng.permutation(len(targets))
            while (perm == np.arange(len(targets))).any(): perm = rng.permutation(len(targets))
            donor = [targets[j] for j in perm]
        for i, p in enumerate(targets):
            if self.use_trans and cw is not None and p in self.parts:
                num = np.zeros(len(self.G.genes)); den = np.zeros(len(self.G.genes))
                for n, wc, y, ok in self.parts[p]:
                    num += cw[n] * wc * y; den += cw[n] * wc * ok
                out[i] = self.alpha * num / (den + self.lam * (sum(cw.values()) / len(cw)))
            elif self.use_trans and donor[i] in self.sig:
                sg = self.sig[donor[i]]
                if self.topk > 0:
                    w = np.full(len(sg), self.alpha_rest if self.alpha_rest is not None else 0.0, np.float32)
                    w[np.argsort(-np.abs(sg))[:self.topk]] = self.alpha
                    out[i] = w * sg
                else:
                    out[i] = self.alpha * sg
            if self.generic > 0: out[i] += self.generic * self.gen
            if self.use_cis:
                g, b = self.G.cis(p); out[i, g] += self.own * self.beta[b]
            k = self.G.idx.get(p)
            if k is not None: out[i, k] = self.own
        return out


def score_space(lnr, c):
    """Predicted delta in the scorer's space for a context with controls c (pseudobulk approximation)."""
    xc = np.expm1(c.ctrl.astype(np.float64)); f = np.exp(lnr) * xc[None]
    f = f / f.sum(1, keepdims=True) * xc.sum()
    return (np.log1p(f) - np.log1p(xc)[None]).astype(np.float32)


def metrics(pred, real, c, perts):
    """Pseudobulk proxies: pds-like (cosine rank, 0.5 = chance), mse skill vs the true context mean, median r."""
    xc = np.expm1(c.ctrl)
    gi = {g: k for k, g in enumerate(c.genes)}
    tg = [gi[p] for p in perts if p in gi]
    keep = c.measured & (xc >= X_MIN); keep[tg] = False
    R = np.nan_to_num(real[:, keep]); P = np.nan_to_num(pred[:, keep])
    Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    D = 1 - Pn @ Rn.T                                   # pred i vs real j
    n = len(perts)
    own = np.diag(D)[:, None]
    rank = (D < own).sum(1) + 0.5 * ((D == own).sum(1) - 1)
    pds = 1 - rank / max(n - 1, 1)
    mean = R.mean(0, keepdims=True)
    mse_skill = 1 - ((P - R) ** 2).sum() / max(((mean - R) ** 2).sum(), 1e-9)
    r = [np.corrcoef(P[i], R[i])[0, 1] if P[i].std() > 0 else np.nan for i in range(n)]
    return float(np.mean(pds)), float(mse_skill), float(np.nanmedian(r))


def loco(C, model_kw, groups=GROUPS, targets=None, min_cells=50, quiet=False):
    res = {}
    for grp in groups:
        held = [g for g in grp if g in C]
        if not held: continue
        train = {k: v for k, v in C.items() if k not in grp}
        sels = {h: [j for j, p in enumerate(C[h].perts) if C[h].ncell[j] >= min_cells and (targets is None or p in targets)]
                for h in held}
        m = GeneralModel(**model_kw).fit(train, targets={C[h].perts[j] for h in held for j in sels[h]})
        for h in held:
            c = C[h]
            sel = sels[h]
            if len(sel) < 10: continue
            perts = list(c.perts[sel])
            pred = score_space(m.predict(perts, c.ctrl), c)
            res[h] = metrics(pred, c.delta[sel], c, perts) + (len(perts),)
            if not quiet: print(f"  {h:12s} n={len(perts):5d}  pds {res[h][0]:.3f}  mse_skill {res[h][1]:+.3f}  r {res[h][2]:+.3f}", flush=True)
    return res


if __name__ == "__main__" and sys.argv[1:2] == ["loco"]:
    C = load_all()
    vcc = set(challenge_targets())
    for kw in [dict(use_cis=False, use_trans=False), dict(use_cis=True, use_trans=False),
               dict(use_cis=True, use_trans=True)]:
        print(kw, flush=True)
        loco(C, kw)


def remove_shared(trans, k=10):
    """Remove the mean and the top-k principal components of a panel's trans part (rows = targets): components
    shared by many targets do not help tell targets apart (repro/next_20261019)."""
    t = trans.astype(np.float64); t = t - t.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(t, full_matrices=False)
    return t - (t @ vt[:k].T) @ vt[:k]


def export(model, targets, path, ctrl=None, bulk_path=None, shared_k=10, oc_path=None):
    """Effects file for vccsub/make_effect_submission.py and tools/loco.py effectfile: log2 fold changes.
    ctrl: the target context's control expression in scorer space (needed when sim_tau > 0).
    bulk_path: also write the gene-total effects for render mode "dual" (V6): own + cis + the trans part with the
    panel's mean and top-shared_k principal components removed."""
    lfc = model.predict(list(targets), ctrl) / np.log(2)
    np.savez_compressed(path, targets=np.array(list(targets)), lfc=lfc.astype(np.float32))
    print(f"wrote {path}: {len(targets)} targets")
    if bulk_path or oc_path:
        use = model.use_trans; model.use_trans = False
        oc = model.predict(list(targets), ctrl) / np.log(2); model.use_trans = use
    if oc_path:                                          # own + cis only, for tools/make_bulk_effects.py
        np.savez_compressed(oc_path, targets=np.array(list(targets)), lfc=oc.astype(np.float32))
    if bulk_path:
        bulk = oc + remove_shared(lfc - oc, shared_k)
        np.savez_compressed(bulk_path, targets=np.array(list(targets)), lfc=bulk.astype(np.float32))
        print(f"wrote {bulk_path}: totals with the top-{shared_k} shared trans components removed")


def vcc_context_ctrl(name):
    """Control expression (scorer space) of a challenge context A-F from its control cells."""
    import anndata as ad
    A = ad.read_h5ad(f"{DATA}/context_{name}.h5ad")
    s = np.asarray(A.X.sum(0)).ravel().astype(np.float64)
    return np.log1p(50_000.0 * s / s.sum()).astype(np.float32)


if __name__ == "__main__" and sys.argv[1:2] == ["export"]:
    # python -m vcc.gm export <heldout context> <targets json or 'vcc'> <out.npz> [key=value ...]
    import json, ast
    held, tsrc, out = sys.argv[2:5]
    kw = {k: ast.literal_eval(v) if k not in ("bulk_out", "oc_out") else v for k, v in (x.split("=", 1) for x in sys.argv[5:])}
    bulk_out, shared_k, oc_out = kw.pop("bulk_out", None), int(kw.pop("shared_k", 10)), kw.pop("oc_out", None)
    grp = next((g for g in GROUPS if held in g), [held])
    excl = set(x for x in os.environ.get("VCC_EXCLUDE", "").split(",") if x)     # e.g. H1 for the H1 mock exam (no leakage)
    C = load_all(keep_var=float(kw.get("zgate", 0)) > 0); train = {k: v for k, v in C.items() if k not in grp and k not in excl}
    targets = json.load(open(tsrc))["targets"] if tsrc.endswith(".json") else \
        challenge_targets()
    print(f"fit on {sorted(train)}; {kw}")
    ctrl = C[held].ctrl if held in C else vcc_context_ctrl(held)
    export(GeneralModel(**kw).fit(train, targets=targets), targets, out, ctrl, bulk_path=bulk_out, shared_k=shared_k, oc_path=oc_out)
