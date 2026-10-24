"""Quick feasibility screen (2026-10-19): does co-expression in a context's OWN control cells carry
perturbation-specific signal beyond what transfers from other screens?

    python coexp_screen.py H1 HCT116 KOLF200        (scDFM venv)

For each hold-out set: real per-target delta in the scorer's bulk_lognorm(5e4) space vs the set's NTC cells; candidate
predicted directions per target (genes = all except the set's panel targets):
  trans_A   the existing model's trans part (group A effects, own + cis genes removed)
  full_A    the existing model's full prediction (own + cis + trans), the official-style feature set
  coexp     -slope of each gene on the target in the context's control cells (log1p CP10k, partial on log library
            size), estimated from the stage-3 control cells of that context (no perturbed cells used)
Reports per-target cosine with the real delta (median, fraction > 0) and a pds-like rank score (cosine of predicted
p vs real q, own rank, ties 0.5) for full_A and full_A + w * coexp at a few fixed w (scaled to full_A's norm).
This is a screen on hold-out data, not a tuned result; any candidate is re-tested with the official scorer.
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "/home/zxc29051617/VCC"; SIG = f"{ROOT}/scDFM/repro/signature_20261013"; OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{ROOT}/scDFM"); sys.path.insert(0, SIG)
from vcc.gm import Genome
GENES = pd.read_csv(f"{ROOT}/data/vcc2026/gene_names.csv").gene_name.astype(str).tolist(); G = len(GENES); GI = {g: i for i, g in enumerate(GENES)}
STAGE3 = {"H1": "h1_H1", "HCT116": "xatlas_HCT116", "KOLF200": "kolf_KOLF"}
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)


def bulk(sums):
    tot = sums.sum(1, keepdims=True); return np.log1p(sums * (5e4 / np.maximum(tot, 1)))


def real_deltas(name):
    R = ad.read_h5ad(f"{ROOT}/data/stage3/loco/{name}/real.h5ad"); X = sp.csr_matrix(R.X); lab = R.obs.target_gene.astype(str).values
    targets = json.load(open(f"{ROOT}/data/stage3/loco/{name}/real.json"))["targets"]
    ctrl = np.asarray(X[lab == "non-targeting"].sum(0)).ravel()[None].astype(np.float64)
    S = np.vstack([np.asarray(X[lab == t].sum(0)).ravel() for t in targets]).astype(np.float64)
    mu_c = bulk(ctrl)[0]; D = bulk(S) - mu_c[None]
    rate = ctrl[0] / ctrl.sum()
    return targets, D, rate, mu_c


def pred_deltas(lfc, rate, mu_c):
    fc = 2.0 ** np.clip(lfc, -4, 4); m = rate[None] * fc
    return np.log1p(m * (5e4 / m.sum(1, keepdims=True))) - mu_c[None]


def coexp(name, targets, nmax=20000, seed=0):
    A = ad.read_h5ad(f"{ROOT}/data/stage3/{STAGE3[name]}.h5ad", backed="r")
    idx = np.flatnonzero(A.obs.is_control.values)
    idx = np.sort(np.random.default_rng(seed).choice(idx, min(nmax, len(idx)), replace=False))
    X = sp.csr_matrix(A.X[idx]).astype(np.float64); A.file.close()
    lib = np.asarray(X.sum(1)).ravel(); X = sp.diags(1e4 / lib) @ X; X.data = np.log1p(X.data); X = X.tocsc()
    n = X.shape[0]; z = np.log(lib); z = (z - z.mean()) / z.std()
    mean = np.asarray(X.mean(0)).ravel(); ex2 = np.asarray(X.multiply(X).mean(0)).ravel(); sd = np.sqrt(np.maximum(ex2 - mean ** 2, 0))
    cz = np.asarray(X.T @ z).ravel() / n                                     # cov(gene, z), z standardised
    tk = [GI.get(t) for t in targets]
    out = np.zeros((len(targets), G)); ok_t = np.zeros(len(targets), bool)
    for i, k in enumerate(tk):
        if k is None or sd[k] <= 0: continue
        xk = X[:, k].toarray().ravel()
        cov = np.asarray(X.T @ xk).ravel() / n - mean * mean[k]
        cov_p = cov - cz * cz[k]                                              # partial out log library size
        var_k = sd[k] ** 2 - cz[k] ** 2
        if var_k <= 1e-12: continue
        out[i] = -cov_p / var_k                                               # -slope of gene on target
        out[i, k] = 0; ok_t[i] = True
    return out, ok_t, int(n), sd


def cos_rows(P, D, mask):
    a = P[:, mask]; b = D[:, mask]
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def pds_like(P, D, mask):
    a = P[:, mask]; b = D[:, mask]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12); b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    C = a @ b.T; d = np.diag(C)[:, None]
    n = C.shape[0]; less = (C < d).sum(1); tie = (C == d).sum(1) - 1
    return float(np.mean((less + 0.5 * tie) / (n - 1)))


if __name__ == "__main__":
    rep = {}
    gen = Genome(GENES)
    for name in sys.argv[1:] or ["H1", "HCT116", "KOLF200"]:
        t0 = time.time()
        targets, D, rate, mu_c = real_deltas(name)
        z = np.load(f"{SIG}/effects/{name}_A.npz", allow_pickle=True); assert list(z["targets"]) == targets
        full = pred_deltas(z["lfc"], rate, mu_c)
        panel = np.zeros(G, bool); panel[[GI[t] for t in targets if t in GI]] = True
        mask = ~panel & (rate > 0)
        # trans-only: remove each target's own gene and cis genes from the prediction
        trans = full.copy()
        for i, t in enumerate(targets):
            g, _ = gen.cis(t); trans[i, g] = 0
        CE, okt, ncell, sd = coexp(name, targets)
        r = {"n_targets": len(targets), "control_cells_for_coexp": ncell, "targets_with_coexp": int(okt.sum())}
        for lab, P in [("trans_A", trans), ("full_A", full), ("coexp", CE)]:
            c = cos_rows(P, D, mask)[okt]
            r[f"cos_{lab}_median"] = float(np.median(c)); r[f"cos_{lab}_frac_pos"] = float((c > 0).mean())
        r["pds_like_full_A"] = pds_like(full, D, mask); r["pds_like_coexp"] = pds_like(np.where(okt[:, None], CE, 0), D, mask)
        nf = np.linalg.norm(full[:, mask], axis=1, keepdims=True); nc = np.linalg.norm(CE[:, mask], axis=1, keepdims=True) + 1e-12
        for w in [0.25, 0.5, 1.0]:
            r[f"pds_like_full_A+{w}coexp"] = pds_like(full + w * CE * (nf / nc), D, mask)
        rep[name] = r; log(f"{name} ({time.time() - t0:.0f}s): " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
    json.dump(rep, open(f"{OUT}/coexp_screen.json", "w"), indent=1)
