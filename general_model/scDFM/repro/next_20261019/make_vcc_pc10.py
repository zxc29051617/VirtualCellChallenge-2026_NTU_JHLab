"""A/B/C effects for V5: VCC_gm_vip lfc with the mean and top-10 principal components of the trans part removed
across the 300 challenge targets (own + cis untouched). Checks first that the cache path reproduces VCC_gm_vip."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM"); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/signature_20261013")
import sigexp as S, render_eval as E
from vcc.gm import Genome
E_DIR = "/home/zxc29051617/VCC/data/stage3/effects"
z = np.load(f"{E_DIR}/VCC_gm_vip.npz", allow_pickle=True); targets = [str(t) for t in z["targets"]]; full = z["lfc"].astype(np.float32)
C = S.load_cache(); gen = Genome(S.GENES); w = {c: 1.0 for c in S.CONTEXTS}
mine = S.build_effects(C, S.CONTEXTS, w, targets, gen)
print("cache path vs VCC_gm_vip: max |diff|", float(np.abs(mine - full).max()), "mean |diff|", float(np.abs(mine - full).mean()))
oc = S.build_effects(C, S.CONTEXTS, w, targets, gen, use_trans=False)
trans = full - oc; t2 = E.remove_shared(trans, 10)
print("trans norm before/after (median over targets):", float(np.median(np.linalg.norm(trans, axis=1))), float(np.median(np.linalg.norm(t2, axis=1))))
np.savez_compressed(f"{E_DIR}/VCC_gm_vip_pc10.npz", targets=np.array(targets), lfc=(oc + t2).astype(np.float32))
print("wrote VCC_gm_vip_pc10.npz", len(targets), "targets")
