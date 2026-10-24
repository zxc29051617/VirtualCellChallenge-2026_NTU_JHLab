"""A/B/C gene totals for G1 (2026-10-20): trans part = depth-weighted ridge over all sources (each (source, target) row
weighted ncell / (ncell + 200)), panel mean + top-10 shared components removed, the context's top-50 control-cell axes
removed, per-target magnitudes sharpened (row p x |t_p|^(gamma - 1), gamma 3); totals = own + cis + s_c x trans with s_c
chosen per context so that #348's budget (across-target spread of predicted pseudobulk means) equals the expected claim
(300 x jackknife trace of a 400-cell pseudobulk of that context's controls). Uses only predictions and released controls.
Cells stay VCC_gm_vip (V1)."""
import sys
import numpy as np, scipy.sparse as sp, anndata as ad
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM"); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/signature_20261013")
import render_eval as E, sigexp as S, weights_screen as W, budget_abc as BA
from vcc.gm import Genome
GAMMA = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0; TAG = sys.argv[2] if len(sys.argv) > 2 else "g1"
Ed = "/home/zxc29051617/VCC/data/stage3/effects"
z = np.load(f"{Ed}/VCC_gm_vip.npz", allow_pickle=True); targets = [str(t) for t in z["targets"]]; full = z["lfc"].astype(np.float64)
Ca = S.load_cache()
oc = S.build_effects(Ca, S.CONTEXTS, {c: 1.0 for c in S.CONTEXTS}, targets, Genome(S.GENES), use_trans=False).astype(np.float64)
t_eq = W.trans(Ca, S.CONTEXTS, targets, 0, W.WEIGHTS["equal"]); print("equal-weight trans reproduces VCC_gm_vip:", float(np.abs(oc + t_eq - full).max()), flush=True)
t5 = E.remove_shared(W.trans(Ca, S.CONTEXTS, targets, 0, W.WEIGHTS["depth200"]).astype(np.float32)).astype(np.float64)
own = np.array([S.GI.get(t, -1) for t in targets]); rng = np.random.default_rng(0)
for x in "ABC":
    A = ad.read_h5ad(f"/home/zxc29051617/VCC/data/vcc2026/context_{x}.h5ad"); C = sp.csr_matrix(A.X); lib = np.asarray(C.sum(1)).ravel()
    V = E.control_axes(C, lib, 50).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
    t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** (GAMMA - 1))[:, None]
    Xd = C.astype(np.float64); ref_mean = np.asarray(Xd.sum(0)).ravel() / Xd.shape[0]
    claim = len(targets) * BA.jk_trace(Xd, rng)
    bud = lambda s: BA.budget(BA.bulk_means(oc + s * t8, ref_mean), own)
    lo, hi = 0.0, 1.0
    while bud(hi) < claim: hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if bud(mid) < claim else (lo, mid)
    s_c = hi; tot = oc + s_c * t8
    np.savez_compressed(f"{Ed}/VCC_gm_vip_{TAG}_{x}.npz", targets=np.array(targets), lfc=tot.astype(np.float32))
    e = (t8 ** 2).sum(1); pr = e.sum() ** 2 / (e ** 2).sum()
    print(f"{x}: claim {claim:.0f}, s_c {s_c:.4g}, budget {bud(s_c):.0f}, trans PR {pr:.0f}/{len(targets)}, max |lfc| totals {np.abs(tot).max():.2f}, "
          f"top targets {[targets[i] for i in np.argsort(-e)[:6]]}", flush=True)
