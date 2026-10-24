"""Training data for the trained response model (2026-10-22).

    .venv/bin/python -I repro/model_v2_20261022/build_data.py        (from scDFM/)

For every public context (data/stage3/stats + stats_ext via vcc.contexts) writes data/stage3/model_v2/<context>.npz:
  perts, ncell        perturbation groups with >= 20 cells
  Y                   float16 [perts x 18,533]: ln ratio to the context's controls (vcc.gm._ratios: CP50k, EPS 0.5) minus
                      the context's common mode (mean over its perturbations); NaN where the gene is unmeasured or below
                      X_MIN in the controls. Same y as the current model's fit uses, before own/cis masking.
  var                 float16: sampling variance of each entry (Poisson delta method, vcc.contexts.build)
  own_obs             float32: the target's own ln ratio (knockdown strength), NaN if the target is not a measured gene
  ctrl, measured, cm  the control profile log1p(CP50k) (the cell state the model sees), measured mask, common mode
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "/home/zxc29051617/VCC/scDFM")
from vcc.contexts import CONTEXTS, load, available
from vcc.gm import _ratios

OUT = "/home/zxc29051617/VCC/data/stage3/model_v2"
os.makedirs(OUT, exist_ok=True)
t0 = time.time(); total = 0
for name in available():
    c = load(name, min_cells=20, keep_var=True)
    L, xc = _ratios(c)
    cm = np.nanmean(L, 0).astype(np.float32)
    Y = (L - cm[None]).astype(np.float16)
    gi = {g: k for k, g in enumerate(c.genes)}
    own = np.array([L[j, gi[p]] if p in gi else np.nan for j, p in enumerate(c.perts)], np.float32)
    np.savez(f"{OUT}/{name}.npz", perts=c.perts.astype(str), ncell=c.ncell, Y=Y, var=c.var.astype(np.float16), own_obs=own,
             ctrl=c.ctrl.astype(np.float32), measured=c.measured, cm=cm, genes=c.genes.astype(str))
    total += len(c.perts)
    fin = np.isfinite(L).mean()
    print(f"{name:13s} {len(c.perts):6d} perts, {fin:.2f} of entries usable, knockdown median {np.nanmedian(own):+.2f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del c, L, Y
print(f"{total} (context, target) pairs written to {OUT}")
