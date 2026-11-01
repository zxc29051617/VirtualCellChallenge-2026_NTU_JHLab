"""Real perturbed groups vs equal-size control draws: jackknife trace of the pseudobulk (bulk_lognorm 5e4). (2026-10-20)"""
import sys, json
import numpy as np, scipy.sparse as sp, anndata as ad
sys.path.insert(0, ".")
import budget_abc as BA
ROOT = "/home/zxc29051617/VCC"
rng = np.random.default_rng(0)
for name in ["H1", "KOLF200", "HCT116"]:
    R = ad.read_h5ad(f"{ROOT}/data/stage3/loco/{name}/real.h5ad"); X = sp.csr_matrix(R.X).astype(np.float64); lab = R.obs.target_gene.astype(str).values
    targets = json.load(open(f"{ROOT}/data/stage3/loco/{name}/real.json"))["targets"]
    C = X[lab == "non-targeting"]
    vp, vc, n_ = [], [], []
    for t in targets[:120]:
        Y = X[lab == t]; n = Y.shape[0]
        if n < 50 or n > C.shape[0] // 2: continue
        vp.append(BA.jk_trace(Y, rng, n=n, reps=1)); vc.append(BA.jk_trace(C, rng, n=n, reps=3)); n_.append(n)
    vp, vc = np.array(vp), np.array(vc)
    print(f"{name}: {len(vp)} targets, cells/target median {np.median(n_):.0f}; jackknife trace perturbed / control draw: median ratio {np.median(vp / vc):.3f} (IQR {np.percentile(vp / vc, 25):.3f}-{np.percentile(vp / vc, 75):.3f})", flush=True)
