"""A/B/C gene-total effects for V8: VCC_gm_vip_pc10 trans part (shared components removed) with each context's top-50
control-cell axes also removed (axes from that context's own released control cells). Cells stay VCC_gm_vip."""
import sys, numpy as np, scipy.sparse as sp, anndata as ad
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM"); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/signature_20261013")
import render_eval as E, sigexp as S
from vcc.gm import Genome
Ed = "/home/zxc29051617/VCC/data/stage3/effects"
z = np.load(f"{Ed}/VCC_gm_vip.npz", allow_pickle=True); targets = [str(t) for t in z["targets"]]; full = z["lfc"].astype(np.float64)
oc = S.build_effects(S.load_cache(), S.CONTEXTS, {c: 1.0 for c in S.CONTEXTS}, targets, Genome(S.GENES), use_trans=False).astype(np.float64)
t5 = E.remove_shared((full - oc).astype(np.float32)).astype(np.float64)
for x in "ABC":
    A = ad.read_h5ad(f"/home/zxc29051617/VCC/data/vcc2026/context_{x}.h5ad"); C = sp.csr_matrix(A.X); lib = np.asarray(C.sum(1)).ravel()
    V = E.control_axes(C, lib, 50).astype(np.float64)
    t8 = t5 - (t5 @ V.T) @ V
    np.savez_compressed(f"{Ed}/VCC_gm_vip_v8_{x}.npz", targets=np.array(targets), lfc=(oc + t8).astype(np.float32))
    print(x, "wrote; trans norm kept", round(float(np.linalg.norm(t8) / np.linalg.norm(t5)), 3), flush=True)
