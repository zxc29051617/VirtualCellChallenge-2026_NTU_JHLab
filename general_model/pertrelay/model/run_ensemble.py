"""Seed ensemble of AdaptNet (beta 1): average of 3 models trained with different seeds, hidden-cell-type benchmark."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B, adaptnet as AN
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]
_C = {}
def ens(train, held, targets):
    key = tuple(sorted(train))
    if key not in _C:
        _C.clear(); torch.cuda.empty_cache()
        _C[key] = [AN.train(list(train), steps=1500, k=200, nce=1.0, seed=s, log=False, beta_fix=1.0) for s in SEEDS]
    return np.mean([AN.predict(m, held, targets) for m in _C[key]], 0)
B.run(sys.argv[1], ens)
