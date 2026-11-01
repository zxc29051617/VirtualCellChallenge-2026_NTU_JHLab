"""Joint check around V8 with the exact pds proxy: shared components k x control axes m, and cis weight."""
import json, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay"); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013")
import coexp_screen as CS, render_eval as E, sigexp as S
from vcc.gm import Genome
gen = Genome(S.GENES)
def rm(t, k):
    t = t - t.mean(0, keepdims=True)
    if k == 0: return t
    _, _, vt = np.linalg.svd(t, full_matrices=False); return t - (t @ vt[:k].T) @ vt[:k]
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); _, oc, full, shuf = E.components(name); tr = (full - oc).astype(np.float64)
    D0, R, C, lib, rate0, ref_mean = E.load_set(name); V = E.control_axes(C, lib, 100).astype(np.float64)
    panel = np.zeros(S.G, bool); panel[[S.GI[t] for t in targets if t in S.GI]] = True; mp = ~panel & (rate > 0)
    sc = lambda lfc: round(CS.pds_like(CS.pred_deltas(lfc, rate, mu_c), D, mp), 4)
    row = {}
    for k in [0, 5, 10, 20]:
        b = rm(tr, k)
        for m in [50, 100]:
            Vm = V[:m]; row[f"k{k}_m{m}"] = sc(oc + (b - (b @ Vm.T) @ Vm))
    b = rm(tr, 10); Vm = V[:50]; b = b - (b @ Vm.T) @ Vm
    o2 = oc.copy()
    for i, t in enumerate(targets):
        gg, _ = gen.cis(t); o2[i, gg] *= 2
    row["k10_m50_cis2"] = sc(o2 + b)
    print(name, json.dumps(row), flush=True)
