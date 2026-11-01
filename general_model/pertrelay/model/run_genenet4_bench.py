"""GeneNet4, hidden-cell-type benchmark only (after the off-axis fix), without ELEM."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import bench as B, genenet as GN
_C = {}
def get(train):
    key = tuple(sorted(train))
    if key not in _C: _C.clear(); torch.cuda.empty_cache(); _C[key] = GN.train(list(train), log=False)
    return _C[key]
B.run(sys.argv[1], lambda tr, h, t: GN.predict(get(tr), h, t))
