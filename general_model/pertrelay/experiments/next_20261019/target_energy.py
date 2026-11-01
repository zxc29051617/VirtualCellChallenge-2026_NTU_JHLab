"""Which targets carry the real signal (noise-corrected energy) and how well do we predict them? (2026-10-20, analysis)"""
import sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, coexp_screen as CS
for name in ["H1", "KOLF200", "HCT116"]:
    B = BE.load(name); D = B["D"].copy(); oc = B["oc"]; tr = B["full"] - oc
    Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    t5 = E.remove_shared(tr).astype(np.float64); t8 = t5 - (t5 @ Vc.T) @ Vc
    Q = CS.pred_deltas(oc + 0.6 * t8, B["rate"], B["mu_c"])
    for i, k in enumerate(B["own"]):
        if k >= 0: D[i, k] = 0; Q[i, k] = 0
    dist = (D ** 2).sum(1) - B["v"]; qd = (Q * D).sum(1); qq = (Q ** 2).sum(1)
    o = np.argsort(-dist); cum = np.cumsum(dist[o]) / dist.sum()
    n = len(dist)
    cosp = qd / np.sqrt(np.maximum(qq * np.maximum(dist, 1e-9), 1e-12))
    print(f"{name}: {n} targets; energy share of top 5/10/20/50: {cum[4]:.2f}/{cum[9]:.2f}/{cum[19]:.2f}/{cum[49]:.2f}; "
          f"targets with dist<=0: {(dist <= 0).sum()}; share of sum q.r from top-20 energy targets: {qd[o[:20]].sum() / qd.sum():.2f}")
    print("   top-12 energy targets: " + "  ".join(f"{B['targets'][i]}(e{dist[i]/dist.sum()*100:.1f}% cos{cosp[i]:+.2f} q{np.sqrt(qq[i]):.1f}/r{np.sqrt(max(dist[i],0)):.1f})" for i in o[:12]))
    # rank correlation between predicted magnitude and true energy
    from scipy.stats import spearmanr
    print(f"   spearman(|q_p|, true energy) {spearmanr(np.sqrt(qq), dist).correlation:.3f}; median cos_p top-20 {np.median(cosp[o[:20]]):.3f}, rest {np.median(cosp[o[20:]]):.3f}")
