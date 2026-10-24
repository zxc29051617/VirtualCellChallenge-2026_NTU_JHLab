"""Natural variation in the new cell type's own control cells as a knockdown predictor (2026-10-23).

    RPE1=1 .venv/bin/python -I repro/model_v2_20261022/natvar.py [held] [n_pcs]        (from scDFM/; default H1, 10)

Idea: in ~20,000 unperturbed cells, cells that happen to express target X lower show what lower X does in THIS cell
type. For each benchmark target X (expressed in the controls): partial correlation of every gene with X across the
control cells after removing library size and the top n_pcs cell-state axes; predicted knockdown response
= -corr x sd(gene) (log1p CP10k units). Scored on the hidden-cell-type benchmark targets of `held` (bench.score: dir,
pds, skill), alone and blended with AdaptNet trained without `held`'s group.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp, anndata as ad, torch
import bench as B

held = sys.argv[1] if len(sys.argv) > 1 else "H1"; NPC = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SRC = {"H1": "/home/zxc29051617/VCC/data/stage3/h1_H1.h5ad"}
A = ad.read_h5ad(SRC[held], backed="r"); ci = np.flatnonzero(A.obs["is_control"].values.astype(bool))
X = sp.csr_matrix(A[ci].X).astype(np.float32); del A
lib = np.asarray(X.sum(1)).ravel(); N = sp.diags((1e4 / lib).astype(np.float32)) @ X
mu = np.asarray(N.mean(0)).ravel(); ok = np.flatnonzero(mu > 0.05)
L = N[:, ok].tocsr(); L.data = np.log1p(L.data); L = L.toarray(); L -= L.mean(0)
print(f"{held}: {L.shape[0]} control cells, {L.shape[1]} expressed genes", flush=True)
# confounders: log library size + top NPC state axes (randomised SVD on a cell subsample for the gene loadings)
Lt = torch.tensor(L, device="cuda")
U, S, V = torch.svd_lowrank(Lt, q=NPC, niter=4)
Zc = torch.cat([torch.tensor(np.log(lib) - np.log(lib).mean(), device="cuda", dtype=torch.float32)[:, None], U], 1)
Q, _ = torch.linalg.qr(Zc)
Lr = Lt - Q @ (Q.T @ Lt)                                   # residual expression
sd = Lr.std(0) + 1e-6
col = {int(g): j for j, g in enumerate(ok)}

train = [c for c in B.CONTEXTS if c not in [f for f in B.FOLDS.values() if held in f][0]]
targets = B.eval_targets(held, train)
gi = B.genome().idx; G = len(B.genome().genes)
P = np.zeros((len(targets), G), np.float32); have = 0
for i, p in enumerate(targets):
    j = col.get(gi.get(p, -1))
    if j is None: continue
    x = Lr[:, j]; x = (x - x.mean()) / (x.std() + 1e-6)
    corr = (Lr.T @ x) / Lr.shape[0] / sd
    out = np.zeros(G, np.float32); out[ok] = (-corr * sd).cpu().numpy(); out[gi[p]] = 0.0
    P[i] = out; have += 1
print(f"{have}/{len(targets)} targets expressed in the controls", flush=True)
r = B.score(held, targets, P); print(f"natural variation alone: dir {r['dir']:+.4f} pds {r['pds']:.4f} skill {r['skill']:+.4f}", flush=True)

import adaptnet as AN
m = AN.train(train, steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0)
Pa = AN.predict(m, held, targets).astype(np.float64)
ra = B.score(held, targets, Pa.astype(np.float32)); print(f"AdaptNet:                dir {ra['dir']:+.4f} pds {ra['pds']:.4f} skill {ra['skill']:+.4f}", flush=True)
na = np.linalg.norm(Pa, axis=1, keepdims=True) + 1e-12; nn_ = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
for w in (0.1, 0.2, 0.35, 0.5):
    Pb = (Pa / na * (1 - w) + P / nn_ * w) * na
    rb = B.score(held, targets, Pb.astype(np.float32)); print(f"blend w {w}:            dir {rb['dir']:+.4f} pds {rb['pds']:.4f} skill {rb['skill']:+.4f}", flush=True)
