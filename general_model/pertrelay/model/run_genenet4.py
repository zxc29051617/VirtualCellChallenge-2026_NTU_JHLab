"""GeneNet4 (frozen AdaptNet expert for measured genes + knowledge branch, size calibration) on both benchmarks.
    run_genenet3.py <name>        (set ELEM=1 to add ELEM-1200RS as training-only contexts)"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B, bench_genes as BG, genenet as GN
name = sys.argv[1]; KW = {}
_C = {}
def get(train, drop=()):
    key = (tuple(sorted(train)), hash(frozenset(drop)))
    if key not in _C:
        _C.clear(); torch.cuda.empty_cache(); _C[key] = GN.train(list(train), drop_targets=drop, log=False, **KW)
    return _C[key]
if "--s2-only" not in sys.argv: B.run(name, lambda tr, h, t: GN.predict(get(tr), h, t))
gf = BG.gene_folds()
def m_genenet(train_rows, c, tg, st):
    fold = gf[tg[0]]; drop = [p for p, f in gf.items() if f == fold]
    return GN.predict(get(list(train_rows), drop), c, tg)
BG.run(name, m_genenet, settings=("S2",))
