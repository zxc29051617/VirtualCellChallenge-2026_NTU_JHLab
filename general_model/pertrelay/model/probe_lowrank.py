"""Does a shared gene-program space carry transferable signal? Project the depth-weighted consensus onto the top-k
programs (SVD of the training contexts' responses, per fold) and score on the hidden cell type (bench.py)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bench as B
_V = {}
def programs(train, k):
    key = tuple(sorted(train))
    if key not in _V:
        rng = np.random.default_rng(0); rows = []
        for t in train:
            Y = B.data()[t]["Y"]; idx = rng.choice(len(Y), min(1500, len(Y)), replace=False)
            rows.append(np.nan_to_num(Y[idx].astype(np.float32)))
        X = torch.tensor(np.vstack(rows), device="cuda")
        _, _, V = torch.svd_lowrank(X, q=500, niter=4); _V.clear(); _V[key] = V.cpu().numpy()
    return _V[key][:, :k]
if __name__ == "__main__":
  for k in [20, 50, 100, 200, 500]:
      def m(train, held, targets, k=k):
          C = B.consensus(train, held, targets, n0=200.0); V = programs(train, k); return (C @ V) @ V.T
      B.run(f"cons_d200_proj{k}", m, quiet=True)
      import json; r = json.load(open(f"{B.HERE}/results/cons_d200_proj{k}.json"))["overall"]
      print(f"k {k:4d}: dir {r['dir']:+.4f}  pds {r['pds']:.4f}  skill {r['skill']:+.4f}  scale {r['scale']:.2f}", flush=True)
