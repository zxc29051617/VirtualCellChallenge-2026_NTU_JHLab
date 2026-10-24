"""Is our cells' pseudobulk jackknife trace at least the real one's? (2026-10-20) The #348 claim is sum_p min(pred_jk,
real_jk); the budget = claim scale assumes pred_jk >= real_jk (~ a 400-cell control draw's trace). Renders 30 targets of
context A with the G3c settings (cells VCC_gm_vip_c12, totals VCC_gm_vip_g3a_A, dual, top 2 %) and compares traces."""
import sys
import numpy as np, scipy.sparse as sp, anndata as ad
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM")
import budget_abc as BA
import vcc.render as VR
from vcc.render import render
VR.TOP_FRAC = 0.02
E = "/home/zxc29051617/VCC/data/stage3/effects"
A = ad.read_h5ad("/home/zxc29051617/VCC/data/vcc2026/context_A.h5ad"); C = sp.csr_matrix(A.X).astype(np.float64)
lib = np.asarray(C.sum(1)).ravel(); rate = np.asarray(C.sum(0)).ravel() / lib.sum(); ref_mean = np.asarray(C.sum(0)).ravel() / C.shape[0]
cells = np.load(f"{E}/VCC_gm_vip_c12.npz", allow_pickle=True); tot = np.load(f"{E}/VCC_gm_vip_g3a_A.npz", allow_pickle=True)
rng = np.random.default_rng(0); ctrl = []; pred = []
idx = rng.choice(len(cells["targets"]), 30, replace=False)
for i in idx:
    draw = rng.choice(C.shape[0], 400, replace=False)
    Y = render(C[draw], lib[draw], rate, ref_mean, cells["lfc"][i].astype(np.float64), np.random.default_rng(i), mode="dual",
               lfc_bulk=tot["lfc"][i].astype(np.float64), fix_seed=i).astype(np.float64)
    pred.append(BA.jk_trace(sp.csr_matrix(Y), rng, n=400, reps=1)); ctrl.append(BA.jk_trace(C[draw], rng, n=400, reps=1))
pred, ctrl = np.array(pred), np.array(ctrl)
print(f"30 targets of A: pred jackknife trace median {np.median(pred):.2f} (IQR {np.percentile(pred, 25):.2f}-{np.percentile(pred, 75):.2f}); "
      f"control draw {np.median(ctrl):.2f}; ratio pred/ctrl median {np.median(pred / ctrl):.3f}, min {np.min(pred / ctrl):.3f}")
