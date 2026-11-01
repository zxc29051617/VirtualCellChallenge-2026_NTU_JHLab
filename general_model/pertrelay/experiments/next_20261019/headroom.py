"""Headroom check (2026-10-20, oracle, analysis only): global noise-corrected cosine if each target's trans magnitude
were set optimally (c_p = max(0, u_p . r_p) for the unit direction u_p in scorer space), directions unchanged; vs G1."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W, coexp_screen as CS
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
for name in ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]; D = B["D"].copy()
    for i, k in enumerate(B["own"]):
        if k >= 0: D[i, k] = 0
    dist = (D ** 2).sum(1) - B["v"]
    V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    t5 = E.remove_shared(W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]).astype(np.float32)).astype(np.float64)
    t8 = t5 - (t5 @ V.T) @ V
    Q = CS.pred_deltas(oc + 0.05 * t8, B["rate"], B["mu_c"]) - CS.pred_deltas(oc, B["rate"], B["mu_c"])   # trans part in scorer space (small-signal)
    Q0 = CS.pred_deltas(oc, B["rate"], B["mu_c"])
    for i, k in enumerate(B["own"]):
        if k >= 0: Q[i, k] = 0; Q0[i, k] = 0
    U = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    proj = (U * (D - Q0)).sum(1)                                             # optimal per-target magnitude of the trans part
    cos = lambda P: (P * D).sum() / np.sqrt((P ** 2).sum() * dist.sum())
    g1 = U * (np.linalg.norm(t8, axis=1) ** 3)[:, None]; g1 *= np.sqrt(((Q) ** 2).sum()) / np.sqrt((g1 ** 2).sum())
    best_g1 = max(cos(Q0 + a * g1) for a in [0.25, 0.5, 1, 2, 4, 8])
    orac = Q0 + U * np.maximum(proj, 0)[:, None]
    flat = U * 1.0; best_flat = max(cos(Q0 + a * flat * np.linalg.norm(Q) / np.sqrt(len(U))) for a in [0.25, 0.5, 1, 2, 4, 8])
    print(f"{name}: cos with equal per-target magnitudes {best_flat:+.4f} | G1 sharpening {best_g1:+.4f} | oracle magnitudes {cos(orac):+.4f} | "
          f"per-target direction cos (median) {np.median(proj / np.sqrt(np.maximum(dist, 1e-9))):+.3f}", flush=True)
