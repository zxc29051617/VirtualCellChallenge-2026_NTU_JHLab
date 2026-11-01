"""#348 budget vs claim on the challenge contexts A/B/C for candidate gene totals (2026-10-20; no truth needed).

    python budget_abc.py          (pertrelay venv)

claim ~ sum_p min(pred_jk_p, real_jk_p): the jackknife trace of a 400-cell pseudobulk (bulk_lognorm 5e4), estimated from
random 400-cell draws of each context's control cells (real perturbed cells are of the same kind; own gene excluded
is negligible). budget = across-perturbation centred sum of squares of the predicted pseudobulk means (exact totals:
log1p(5e4 x ref_mean x fc / sum)), own target coordinate excluded. Ratio < 1 -> the prediction-side correction is
scaled down by the ratio.
"""
import sys
import numpy as np, scipy.sparse as sp, anndata as ad
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay"); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013")
import sigexp as S
from vcc.gm import Genome
EFF = "/home/zxc29051617/VCC/data/stage3/effects"; DATA = "/home/zxc29051617/VCC/data/vcc2026"


def jk_trace(X, rng, n=400, reps=5):
    out = []
    for _ in range(reps):
        Y = X[rng.choice(X.shape[0], n, replace=False)]
        s = np.asarray(Y.sum(0)).ravel(); T = s.sum(); f = s / T; L = np.asarray(Y.sum(1)).ravel()
        x2 = np.asarray(Y.multiply(Y).sum(0)).ravel(); xl = np.asarray(Y.T @ L).ravel()
        varf = np.maximum(x2 - 2 * f * xl + f ** 2 * (L ** 2).sum(), 0) / T ** 2
        out.append(((5e4 ** 2) * varf / (1 + 5e4 * f) ** 2).sum())
    return float(np.mean(out))


def bulk_means(lfc, ref_mean):
    m = ref_mean[None] * 2.0 ** np.clip(lfc, -4, 4); return np.log1p(m * (5e4 / m.sum(1, keepdims=True)))


def budget(mu, own):
    dev = mu - mu.mean(0, keepdims=True); tot = (dev ** 2).sum()
    for i, k in enumerate(own):
        if k >= 0: tot -= dev[i, k] ** 2
    return tot


if __name__ == "__main__":
    zf = np.load(f"{EFF}/VCC_gm_vip.npz", allow_pickle=True); targets = [str(t) for t in zf["targets"]]
    full = zf["lfc"].astype(np.float64); oc = S.build_effects(S.load_cache(), S.CONTEXTS, {c: 1.0 for c in S.CONTEXTS}, targets, Genome(S.GENES), use_trans=False).astype(np.float64)
    own = np.array([S.GI.get(t, -1) for t in targets])
    rng = np.random.default_rng(0)
    for c in ["A", "B", "C"]:
        A = ad.read_h5ad(f"{DATA}/context_{c}.h5ad"); X = sp.csr_matrix(A.X).astype(np.float64)
        ref_mean = np.asarray(X.sum(0)).ravel() / X.shape[0]
        v = jk_trace(X, rng); claim = len(targets) * v
        f9 = np.load(f"{EFF}/VCC_gm_vip_v9_{c}.npz", allow_pickle=True)["lfc"].astype(np.float64)
        v8 = np.load(f"{EFF}/VCC_gm_vip_v8_{c}.npz", allow_pickle=True)["lfc"].astype(np.float64)
        t8 = v8 - oc
        rows = {"V6 (x1)": np.load(f"{EFF}/VCC_gm_vip_pc10.npz", allow_pickle=True)["lfc"], "V7a": np.load(f"{EFF}/VCC_gm_vip_pc10_s050.npz", allow_pickle=True)["lfc"],
                "F9 = V9": f9, "oc + 0.5 t8": oc + 0.5 * t8, "oc + 0.3 t8": oc + 0.3 * t8, "oc + 0.2 t8": oc + 0.2 * t8, "oc only": oc}
        print(f"context {c}: claim ~ {claim:.1f} (per-target jackknife trace {v:.3f}, median lib {np.median(np.asarray(X.sum(1)).ravel()):.0f})", flush=True)
        for k, lf in rows.items():
            b = budget(bulk_means(np.asarray(lf, np.float64), ref_mean), own)
            print(f"   {k:12s} budget {b:9.1f}  ratio {b / claim:6.3f}", flush=True)
