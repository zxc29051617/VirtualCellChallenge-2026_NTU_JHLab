"""Quick GeneNet check: hidden cell type H1 (seen genes, bench.py setting) and hidden genes fold 0 in H1 (bench_genes S2)."""
import os, sys, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B, bench_genes as BG, genenet as GN
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
held = "H1"; train = [c for c in B.CONTEXTS if c not in B.FOLDS["H1"]]
t0 = time.time(); m = GN.train(train, steps=steps); print(f"trained (seen genes) in {time.time() - t0:.0f}s", flush=True)
tg = B.eval_targets(held, train)
for nm, P in (("consensus_d200", B.consensus(train, held, tg, n0=200.0)), ("genenet", GN.predict(m, held, tg))):
    r = B.score(held, tg, P); print(f"  hidden cell type H1, seen genes: {nm:15s} dir {r['dir']:+.4f} pds {r['pds']:.4f} skill {r['skill']:+.4f}", flush=True)
del m; torch.cuda.empty_cache()
train_rows, tests = BG.setup(0, "H1"); fold_genes = [p for p, f in BG.gene_folds().items() if f == 0]
t0 = time.time(); m = GN.train(train, drop_targets=fold_genes, steps=steps); print(f"trained (fold-0 genes removed) in {time.time() - t0:.0f}s", flush=True)
for c, tgs in tests.items():
    for nm, P in (("blend", BG.m_blend(train_rows, c, tgs, "S2")), ("genenet", GN.predict(m, c, tgs))):
        r = B.score(c, tgs, P); print(f"  hidden genes in hidden {c}: {nm:8s} dir {r['dir']:+.4f} pds {r['pds']:.4f} skill {r['skill']:+.4f}", flush=True)
