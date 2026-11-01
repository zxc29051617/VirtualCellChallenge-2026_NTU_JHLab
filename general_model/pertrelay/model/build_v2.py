"""D/E/F inputs for the hybrid that won the mock exam (2026-10-23): AdaptNet (beta 1) trained on all 14 public contexts
predicts the gene totals of each final context from its own controls; the cells stay those of gm_final_v1.

    .venv/bin/python -I model/build_v2.py        (from pertrelay/)

Writes data/stage3/effects/gm_final_v2pre{.npz, _oc.npz, _d200_D/E/F.npz} for tools/final_pipeline.sh PRECOMP, and the
trained weights to data/stage3/model_v2/adaptnet_all14_seed0.pt.
"""
import os, sys, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else "v2"          # output gm_final_<TAG>pre*; set RPE1=1 etc. to add training contexts
import numpy as np, torch, anndata as ad, scipy.sparse as sp
import bench as B, adaptnet as AN

E = "/home/zxc29051617/VCC/data/stage3/effects"; DATA = "/home/zxc29051617/VCC/data/vcc2026_final"
t0 = time.time()
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]      # SEEDS=0,1,2: average of several trained models
models = []
for sd in SEEDS:
    model = AN.train(list(B.CONTEXTS), steps=1500, k=200, nce=1.0, seed=sd, log=True, beta_fix=1.0, store=models[0][1] if models else None)
    net, S, Z = model
    torch.save({"state_dict": net.state_dict(), "contexts": S.ctx_names, "settings": dict(steps=1500, k=200, nce=1.0, seed=sd, beta_fix=1.0)},
               f"/home/zxc29051617/VCC/data/stage3/model_v2/adaptnet_{TAG}_seed{sd}.pt")
    models.append(model)
    print(f"seed {sd}: trained on {len(S.ctx_names)} contexts ({', '.join(S.ctx_names)}) in {time.time() - t0:.0f}s", flush=True)

zc = np.load(f"{E}/gm_final_v1.npz", allow_pickle=True); zo = np.load(f"{E}/gm_final_v1_oc.npz", allow_pickle=True); zd = np.load(f"{E}/gm_final_v1_d200.npz", allow_pickle=True)
targets = [str(t) for t in zc["targets"]]; oc = zo["lfc"].astype(np.float64)
ref_norm = np.linalg.norm(zd["lfc"].astype(np.float64) - oc)
G = B.genome(); pre = f"{E}/gm_final_{TAG}pre"
for c in "DEF":
    X = sp.csr_matrix(ad.read_h5ad(f"{DATA}/context_{c}.h5ad").X); s = np.asarray(X.sum(0)).ravel(); ctrl = np.log1p(5e4 * s / s.sum())
    T = np.mean([AN.predict_ctrl(m, ctrl, targets).astype(np.float64) for m in models], 0) / np.log(2)
    for i, p in enumerate(targets):
        g, _ = G.cis(p); T[i, g] = 0.0
        if p in G.idx: T[i, G.idx[p]] = 0.0
    kappa = ref_norm / max(np.linalg.norm(T), 1e-12)
    np.savez_compressed(f"{pre}_d200_{c}.npz", targets=np.array(targets), lfc=(oc + kappa * T).astype(np.float32))
    nz = (np.abs(T).sum(1) > 0).sum()
    print(f"{c}: AdaptNet totals input for {nz}/{len(targets)} targets (kappa {kappa:.3f})", flush=True)
np.savez_compressed(f"{pre}.npz", targets=np.array(targets), lfc=zc["lfc"])          # cells input = gm_final_v1's
np.savez_compressed(f"{pre}_oc.npz", targets=np.array(targets), lfc=zo["lfc"])
print(f"wrote {pre}*.npz", flush=True)
