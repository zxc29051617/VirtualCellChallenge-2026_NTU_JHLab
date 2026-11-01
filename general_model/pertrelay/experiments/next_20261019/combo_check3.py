"""Where does per-target sharpening peak, and how concentrated does it make the totals? (2026-10-20, cell-free)
Participation ratio PR = (sum_p e_p)^2 / sum_p e_p^2 (effective number of targets carrying the energy e_p)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
pr = lambda e: e.sum() ** 2 / (e ** 2).sum()
for name in ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]; D = B["D"].copy()
    for i, k in enumerate(B["own"]):
        if k >= 0: D[i, k] = 0
    true_e = np.maximum((D ** 2).sum(1) - B["v"], 0)
    Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    prep = lambda t: (lambda t5: t5 - (t5 @ Vc.T) @ Vc)(E.remove_shared(t).astype(np.float64))
    bn = np.linalg.norm(prep(B["full"] - oc))
    d2 = prep(W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]))
    sharpen = lambda t, g: t * (np.maximum(np.linalg.norm(t, axis=1), 1e-9) ** (g - 1))[:, None]
    out = [f"true PR {pr(true_e):.0f}/{len(targets)}"]
    for g in [1, 2, 3, 4, 6, 8]:
        t = sharpen(d2, g); e = (t ** 2).sum(1)
        r = BE.score(B, oc + 0.6 * t * (bn / np.linalg.norm(t))); out.append(f"g{g}: cos {r['cos']:+.4f} pds {r['pds']:.4f} PR {pr(e):.0f}")
    print(name, " | ".join(out), flush=True)
