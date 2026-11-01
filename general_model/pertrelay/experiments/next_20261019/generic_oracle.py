"""Trade-off check (oracle, analysis only): add gamma x the TRUE panel-mean real delta to the current-style prediction
(own + cis + 0.3 x V8 trans) in scorer space; how do pds and mse_free move? (2026-10-20)"""
import sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, coexp_screen as CS
for name in ["H1", "KOLF200", "HCT116"]:
    B = BE.load(name); D = B["D"]; oc = B["oc"]; tr = B["full"] - oc
    Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    t5 = E.remove_shared(tr).astype(np.float64); t8 = t5 - (t5 @ Vc.T) @ Vc
    Q0 = CS.pred_deltas(oc + 0.3 * t8, B["rate"], B["mu_c"]); g = D.mean(0)
    fin = np.ones_like(B["v"])
    out = []
    for gam in [0.0, 0.25, 0.5, 1.0]:
        Q = Q0 + gam * g[None]
        err = ((Q - D) ** 2).sum(1); dist = (D ** 2).sum(1)
        for i, k in enumerate(B["own"]):
            if k >= 0: err[i] -= (Q[i, k] - D[i, k]) ** 2; dist[i] -= D[i, k] ** 2
        mse = (err - B["v"]).sum() / (dist - B["v"]).sum()
        out.append(f"x{gam}: pds {CS.pds_like(Q, D, B['mask']):.4f} mse {mse:.4f}")
    print(name, "  ".join(out), flush=True)
