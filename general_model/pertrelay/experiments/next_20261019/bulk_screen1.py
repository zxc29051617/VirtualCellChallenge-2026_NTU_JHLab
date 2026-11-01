"""Screen 1 with the cell-free evaluator (2026-10-20): how mse_free and pds move with the gene totals' construction.

    python bulk_screen1.py [H1 KOLF200 HCT116]          (pertrelay venv)

Calibration rows (known on the leaderboard): V6 (mse member 0, i.e. mse >= 1) and V7a (0.832). Then the scale of the
transferred part with own + cis unscaled (F10 family) vs the whole totals scaled (F9 family). Hold-out truth used for
scoring only (screen); anything chosen is re-checked with the official scorer and the leaderboard.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E

res = {}
for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
    B = BE.load(name); oc, full = B["oc"], B["full"]; tr = full - oc
    t5 = E.remove_shared(tr).astype(np.float64)
    V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
    cand = {"control": np.zeros_like(oc), "own+cis": oc, "V1": full, "V6": oc + t5, "V7a": 0.5 * (oc + t5),
            "V7b": 0.25 * (oc + t5), "V8": oc + t8, "V9(F9)": 0.5 * (oc + t8)}
    for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        cand[f"oc+{s}*t8"] = oc + s * t8; cand[f"{s}*(oc+t8)"] = s * (oc + t8)
    row = {k: BE.score(B, v) for k, v in cand.items()}
    res[name] = row
    print(name, f"(real noise share {B['v'].sum() / ((B['D'] ** 2).sum()):.3f})", flush=True)
    for k, v in row.items(): print(f"  {k:14s} mse_free {v['mse_free']:.4f}  pds {v['pds']:.4f}", flush=True)
json.dump(res, open("bulk_screen1.json", "w"), indent=1)
