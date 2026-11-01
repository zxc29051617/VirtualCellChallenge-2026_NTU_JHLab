"""Stronger per-target sharpening and entry-level sharpening on top of depth200 (2026-10-20, cell-free)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
for name in ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]
    Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    prep = lambda t: (lambda t5: t5 - (t5 @ Vc.T) @ Vc)(E.remove_shared(t).astype(np.float64))
    bn = np.linalg.norm(prep(B["full"] - oc))
    d2 = prep(W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]))
    sharpen = lambda t, g: t * (np.maximum(np.linalg.norm(t, axis=1), 1e-9) ** (g - 1))[:, None]
    ent = lambda t, p: np.sign(t) * np.abs(t) ** p
    cands = [("g2", sharpen(d2, 2.0)), ("g2.5", sharpen(d2, 2.5)), ("g3", sharpen(d2, 3.0)), ("g4", sharpen(d2, 4.0)),
             ("g2+ent1.25", sharpen(ent(d2, 1.25), 2.0)), ("g2+ent1.5", sharpen(ent(d2, 1.5), 2.0)), ("g2+ent0.75", sharpen(ent(d2, 0.75), 2.0))]
    out = []
    for k, t in cands:
        r = BE.score(B, oc + 0.6 * t * (bn / np.linalg.norm(t))); out.append(f"{k}: cos {r['cos']:+.4f} pds {r['pds']:.4f}")
    print(name, " | ".join(out), flush=True)
