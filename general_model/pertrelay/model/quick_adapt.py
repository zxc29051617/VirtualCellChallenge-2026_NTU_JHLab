"""Quick check of AdaptNet on single folds."""
import os, sys, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bench as B, adaptnet as AN
folds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["H1"]
kw = dict(a.split("=") for a in sys.argv[2:]); kw = {k: (int(v) if k in ("steps", "k", "bs") else float(v)) for k, v in kw.items()}
for fold in folds:
    held_list = B.FOLDS[fold]; train = [c for c in B.CONTEXTS if c not in held_list]
    t0 = time.time(); model = AN.train(train, **kw); print(f"[{fold}] trained in {time.time() - t0:.0f}s, options {kw}", flush=True)
    for held in held_list:
        tg = B.eval_targets(held, train)
        for name, P in (("consensus_d200", B.consensus(train, held, tg, n0=200.0)), ("adaptnet", AN.predict(model, held, tg))):
            r = B.score(held, tg, P); print(f"  {held:12s} {name:15s} dir {r['dir']:+.4f}  pds {r['pds']:.4f}  skill {r['skill']:+.4f}  scale {r['scale']:.2f}", flush=True)
