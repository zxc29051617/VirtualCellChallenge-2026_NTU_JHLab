"""A/B/C effects files with depth-weighted sources (2026-10-20): VCC_gm_vip_d200.npz (own + cis + depth-weighted trans x3,
= GeneralModel n0=200) and VCC_gm_vip_oc.npz (own + cis), from the signature cache (equal-weight version reproduces
VCC_gm_vip.npz to 1e-6). Inputs for tools/make_bulk_effects.py --gamma 3 --match-claim."""
import sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay"); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013")
import sigexp as S, weights_screen as W
from vcc.gm import Genome
Ed = "/home/zxc29051617/VCC/data/stage3/effects"
z = np.load(f"{Ed}/VCC_gm_vip.npz", allow_pickle=True); targets = [str(t) for t in z["targets"]]
Ca = S.load_cache()
oc = S.build_effects(Ca, S.CONTEXTS, {c: 1.0 for c in S.CONTEXTS}, targets, Genome(S.GENES), use_trans=False).astype(np.float64)
d200 = oc + W.trans(Ca, S.CONTEXTS, targets, 0, W.WEIGHTS["depth200"])
np.savez_compressed(f"{Ed}/VCC_gm_vip_oc.npz", targets=np.array(targets), lfc=oc.astype(np.float32))
np.savez_compressed(f"{Ed}/VCC_gm_vip_d200.npz", targets=np.array(targets), lfc=d200.astype(np.float32))
print("wrote VCC_gm_vip_oc.npz, VCC_gm_vip_d200.npz")
