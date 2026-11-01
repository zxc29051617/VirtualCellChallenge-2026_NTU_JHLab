"""Full leave-one-cell-type-out benchmark for AdaptNet (fixed settings), one or more seeds."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B, adaptnet as AN
for seed in [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]:
    AN._CACHE.clear()
    B.run(f"adaptnet_s{seed}", lambda tr, h, t, seed=seed: AN.method(tr, h, t, steps=1500, k=200, nce=1.0, seed=seed, log=False))
