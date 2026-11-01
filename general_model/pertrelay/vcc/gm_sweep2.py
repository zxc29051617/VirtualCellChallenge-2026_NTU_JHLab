"""Sweep the context-similarity temperature (sim_tau) of the general model, LOCO on the VCC targets."""
import time
import numpy as np
from vcc.contexts import load_all
from vcc.gm import loco, GROUPS

C = load_all()
vcc = set(open("/home/zxc29051617/VCC/data/vcc2026/pert_counts.csv").read().split()[1:])
groups = [g for g in GROUPS if any(x in C for x in g) and g[0] in ("K562", "HCT116", "HEK293T", "CD4_Rest", "FengNF", "KOLF", "H1")]
for tau in [0.0, 5.0, 20.0, 50.0, 100.0]:
    t = time.time()
    r = loco(C, dict(per_celltype=False, lam=3.0, kcap=1.0, sim_tau=tau), groups=groups, targets=None, quiet=True)
    rv = loco(C, dict(per_celltype=False, lam=3.0, kcap=1.0, sim_tau=tau), groups=groups, targets=vcc, quiet=True)
    f = lambda r: " ".join(f"{k}:{v[0]:.3f}" for k, v in r.items())
    print(f"tau={tau:<6} all-perts mean pds {np.mean([v[0] for v in r.values()]):.4f} | {f(r)}", flush=True)
    print(f"{'':10s} VCC-targets mean pds {np.mean([v[0] for v in rv.values()]):.4f} | {f(rv)}  ({time.time() - t:.0f}s)", flush=True)
