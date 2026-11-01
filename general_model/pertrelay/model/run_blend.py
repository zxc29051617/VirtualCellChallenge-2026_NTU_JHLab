"""Does blending the trained model with the plain consensus help? Hidden-cell-type benchmark (2026-10-23).

    RPE1=1 .venv/bin/python -I model/run_blend.py        (from pertrelay/)

P = AdaptNet (beta 1) + w x kappa x consensus_d200, kappa = ||AdaptNet|| / ||consensus|| over the hidden context's panel
(same overall size), w in 0, 0.25, 0.5, 1. One AdaptNet per fold (seed 0), shared by all w.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B, adaptnet as AN

_M, _P = {}, {}


def pair(train, held, targets):
    key = (tuple(sorted(train)), held)
    if key not in _P:
        tk = tuple(sorted(train))
        if tk not in _M:
            _M.clear(); torch.cuda.empty_cache()
            _M[tk] = AN.train(list(train), steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0)
        A = AN.predict(_M[tk], held, targets).astype(np.float64)
        C = B.consensus(train, held, targets, n0=200.0).astype(np.float64)
        _P[key] = (A, C * np.linalg.norm(A) / max(np.linalg.norm(C), 1e-12))
    return _P[key]


for w in (0.0, 0.25, 0.5, 1.0):
    B.run(f"blend_an_cons_w{w}", lambda train, held, targets, w=w: (lambda A, C: (A + w * C).astype(np.float32))(*pair(train, held, targets)))
