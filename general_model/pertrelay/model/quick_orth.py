"""Consensus vs AdaptNet (orth 0 / 10), each scored raw and after the pipeline's panel shared removal (k 10)."""
import os, sys, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bench as B, adaptnet as AN
folds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["H1", "XAtlas"]
for fold in folds:
    held_list = B.FOLDS[fold]; train = [c for c in B.CONTEXTS if c not in held_list]
    models = {"adapt_b1_nce1": AN.train(train, steps=1500, k=200, nce=1.0, seed=0, log=False, beta_fix=1.0),
              "adapt_b1_nce3": AN.train(train, steps=1500, k=200, nce=3.0, seed=0, log=False, beta_fix=1.0)}
    for held in held_list:
        tg = B.eval_targets(held, train)
        preds = {"consensus_d200": B.consensus(train, held, tg, n0=200.0)}
        preds.update({n: AN.predict(m, held, tg) for n, m in models.items()})
        for n, P in preds.items():
            r0 = B.score(held, tg, P); r1 = B.score(held, tg, B.post_shared(P))
            print(f"{held:9s} {n:15s} raw dir {r0['dir']:+.4f} pds {r0['pds']:.4f} | after shared removal dir {r1['dir']:+.4f} pds {r1['pds']:.4f}", flush=True)
