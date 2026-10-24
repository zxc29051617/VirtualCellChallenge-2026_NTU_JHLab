"""Which totals processing suits the trained model? (2026-10-22, H1 hidden, all 298 benchmark targets, no scorer)

The pipeline turns the predicted trans part into gene totals with steps tuned on the consensus: remove the panel's top-k
shared components, remove 50 control-cell axes, sharpen per target by |t|^(gamma-1), weight targets by (agreement x
n_sources)^0.5. On mock subset s0 these steps raised the consensus's pds (0.835 -> 0.877) but cost the trained model its
lead (0.840 -> 0.853). Screen the steps for both inputs; dir and pds are scale-free (the pipeline rescales to budget = claim).
"""
import os, sys, itertools; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, anndata as ad, scipy.sparse as sp
import bench as B, adaptnet as AN
sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/next_20261019"); sys.path.insert(0, "/home/zxc29051617/VCC/scDFM/repro/signature_20261013")
import render_eval as E
from vcc.gm import remove_shared

train = [c for c in B.CONTEXTS if c not in B.FOLDS["H1"]]; tg = B.eval_targets("H1", train)
C = sp.csr_matrix(ad.read_h5ad("/home/zxc29051617/VCC/data/mock_h1/context_H.h5ad").X).astype(np.float64)
lib = np.asarray(C.sum(1)).ravel(); s = np.asarray(C.sum(0)).ravel(); ctrl = np.log1p(5e4 * s / s.sum())
V = E.control_axes(C, lib, 50).astype(np.float64)
# per-target factor (agreement x n_sources)^0.5 as in tools/target_weights.py, from the training contexts only
D = B.data(); G = B.genome()
f_tw = np.zeros(len(tg))
for i, p in enumerate(tg):
    ys = []
    for t in train:
        j = D[t]["row"].get(p)
        if j is not None: ys.append(np.nan_to_num(D[t]["Y"][j].astype(np.float64) * B.kscale(D[t]["own_obs"][j])))
    if len(ys) > 1:
        Yn = np.stack(ys); Yn = Yn / np.maximum(np.linalg.norm(Yn, axis=1, keepdims=True), 1e-9); cs = Yn @ Yn.T
        a = (cs.sum() - len(ys)) / (len(ys) * (len(ys) - 1))
    else:
        a = 0.0
    f_tw[i] = np.sqrt((max(a, 0) + 1e-3) * len(ys))
f_tw /= np.median(f_tw)

model = AN.train(train, steps=1500, k=200, nce=1.0, seed=0, log=False)
inputs = {"consensus": B.consensus(train, "H1", tg, n0=200.0).astype(np.float64), "trained": AN.predict_ctrl(model, ctrl, tg).astype(np.float64)}
for nm, T in inputs.items():
    for i, p in enumerate(tg):
        g, _ = G.cis(p); T[i, g] = 0.0
        if p in G.idx: T[i, G.idx[p]] = 0.0

def process(T, k, axes, gamma, tw):
    t = remove_shared(T.astype(np.float32), k).astype(np.float64) if k > 0 else T - T.mean(0, keepdims=True)
    if axes: t = t - (t @ V.T) @ V
    if gamma != 1: t = t * (np.maximum(np.linalg.norm(t, axis=1), 1e-9) ** (gamma - 1))[:, None]
    if tw: t = t * f_tw[:, None]
    return t

print(f"{len(tg)} H1 targets; columns: dir  pds", flush=True)
for nm, T in inputs.items():
    r = B.score("H1", tg, T.astype(np.float32)); print(f"{nm:10s} raw input                      dir {r['dir']:+.4f}  pds {r['pds']:.4f}", flush=True)
    for k, axes, gamma, tw in itertools.product([0, 5, 10], [False, True], [1, 2, 3], [False, True]):
        r = B.score("H1", tg, process(T, k, axes, gamma, tw).astype(np.float32))
        tag = f"shared {k:2d} axes {int(axes)} gamma {gamma} tw {int(tw)}"
        print(f"{nm:10s} {tag:30s} dir {r['dir']:+.4f}  pds {r['pds']:.4f}{'   <- pipeline setting' if (k, axes, gamma, tw) == (10, True, 3, True) else ''}", flush=True)
