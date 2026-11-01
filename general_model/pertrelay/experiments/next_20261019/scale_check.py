"""Does shrinking a target's whole predicted change (own + cis + trans, V6 totals) change the pds-like score? (proxy)"""
import sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
import coexp_screen as CS, render_eval as E
for name in ["H1", "KOLF200", "HCT116"]:
    targets, D, rate, mu_c = CS.real_deltas(name); _, oc, full, shuf = E.components(name)
    v6 = oc + E.remove_shared(full - oc)
    panel = np.zeros(len(rate), bool); panel[[CS.GI[t] for t in targets if t in CS.GI]] = True; m = ~panel & (rate > 0)
    print(name, {s: round(CS.pds_like(CS.pred_deltas(s * v6, rate, mu_c), D, m), 4) for s in [1.0, 0.5, 0.25, 0.1]}, flush=True)
