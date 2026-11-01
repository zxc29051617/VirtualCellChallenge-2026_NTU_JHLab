"""Full benchmarks for GeneNet: hidden cell types with measured genes (bench.py, 7 folds) and never-perturbed genes in
hidden cell types (bench_genes.py S2: 5 gene folds x H1 / XAtlas / iPSC)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B, bench_genes as BG, genenet as GN
_C = {}
def get(train, drop=()):
    key = (tuple(sorted(train)), len(drop))
    if key not in _C:
        _C.clear(); torch.cuda.empty_cache(); _C[key] = GN.train(list(train), drop_targets=drop, steps=1500, log=False)
    return _C[key]
B.run("genenet2_elem", lambda tr, h, t: GN.predict(get(tr), h, t))
gf = BG.gene_folds()
def m_genenet(train_rows, c, tg, st):
    fold = gf[tg[0]]; drop = [p for p, f in gf.items() if f == fold]
    return GN.predict(get(list(train_rows), drop), c, tg)
BG.run("genenet2_elem", m_genenet, settings=("S2",))
