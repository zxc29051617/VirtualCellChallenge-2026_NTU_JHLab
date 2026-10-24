"""Leave-one-cell-type-out sweep of the general model on the VCC targets (pseudobulk proxies).

    python -m vcc.gm_sweep
"""
import itertools, time
import numpy as np
from vcc.contexts import load_all
from vcc.gm import loco, GROUPS

C = load_all()
vcc = set(open("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").read().split()[1:])
groups = [g for g in GROUPS if any(x in C for x in g) and g[0] in ("K562", "HCT116", "HEK293T", "CD4_Rest", "FengNF", "KOLF")]
print("held-out groups:", groups, flush=True)
rows = []
for pc, lam, kcap in itertools.product([True, False], [0.3, 1.0, 3.0], [1.0, 3.0]):
    t = time.time()
    r = loco(C, dict(per_celltype=pc, lam=lam, kcap=kcap), groups=groups, targets=vcc, quiet=True)
    pds = np.mean([v[0] for v in r.values()]); rr = np.mean([v[2] for v in r.values()])
    detail = " ".join(f"{k}:{v[0]:.3f}" for k, v in r.items())
    print(f"per_celltype={pc!s:5s} lam={lam:<4} kcap={kcap:<4}  mean pds {pds:.4f}  mean r {rr:+.4f}  | {detail}  ({time.time() - t:.0f}s)", flush=True)
