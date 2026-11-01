"""Programs + part of the detail: yhat = P_k + beta (C - P_k), C = depth-weighted consensus, P_k its projection on the
top-k training programs. Scored on the hidden cell types (bench.py)."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bench as B, probe_lowrank as PL
for k in [100, 200]:
    for beta in [0.25, 0.5, 0.75]:
        def m(train, held, targets, k=k, beta=beta):
            C = B.consensus(train, held, targets, n0=200.0); V = PL.programs(train, k); P = (C @ V) @ V.T
            return P + beta * (C - P)
        B.run(f"mix_k{k}_b{beta}", m, quiet=True); r = json.load(open(f"{B.HERE}/results/mix_k{k}_b{beta}.json"))["overall"]
        print(f"k {k} beta {beta}: dir {r['dir']:+.4f}  pds {r['pds']:.4f}  skill {r['skill']:+.4f}  scale {r['scale']:.2f}", flush=True)
