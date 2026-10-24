"""Quick check of ResponseNet on one fold (H1 hidden), short training."""
import sys, time
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bench as B, responsenet as RN
fold = sys.argv[1] if len(sys.argv) > 1 else "H1"; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
kw = dict(a.split("=") for a in sys.argv[3:]); kw = {k: (float(v) if k in ("nce", "tau", "lr") else v == "1") for k, v in kw.items()}
held_list = B.FOLDS[fold]; train = [c for c in B.CONTEXTS if c not in held_list]
t0 = time.time(); model = RN.train(train, steps=steps, **kw); print("options", kw); print(f"trained in {time.time() - t0:.0f}s", flush=True)
for held in held_list:
    tg = B.eval_targets(held, train)
    for name, P in (("consensus_d200", B.consensus(train, held, tg, n0=200.0)), ("responsenet", RN.predict(model, held, tg))):
        r = B.score(held, tg, P); print(f"  {held:12s} {name:15s} dir {r['dir']:+.4f}  pds {r['pds']:.4f}  skill {r['skill']:+.4f}  scale {r['scale']:.2f}", flush=True)
