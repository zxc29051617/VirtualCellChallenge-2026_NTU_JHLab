"""k (shared components) and m (control axes) re-checked for the refined totals (entry agreement k1 + (a n)^0.5 x |t|^2)
(2026-10-20, cell-free). Approx value 1.94 d(pds) + 0.95 d(cos) vs (k 10, m 50)."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W, energy_proxy_screen as EP, agree_totals_screen as AT
SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"
tot = {}
for name in ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]
    Vall = E.control_axes(B["C"], B["lib"], 100).astype(np.float64)
    bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
    td = W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]); ag = AT.entry_agreement(C, train, targets, td.shape)
    td = td * np.clip(ag / np.median(ag[np.abs(td) > 0]), 0, None)
    a, n = EP.agreement(C, train, targets); f = np.sqrt((np.maximum(a, 0) + 1e-3) * n / n.mean())
    res = {}
    for k in [5, 10, 15]:
        for m in [20, 50, 100]:
            t = td - td.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(t, full_matrices=False); t = t - (t @ vt[:k].T) @ vt[:k]
            V = Vall[:m]; t = t - (t @ V.T) @ V
            t = t * (np.maximum(np.linalg.norm(t, axis=1), 1e-9) ** 2 * f)[:, None]
            res[(k, m)] = BE.score(B, oc + 0.6 * t * (bn / np.linalg.norm(t)))
    ref = res[(10, 50)]
    out = []
    for key, r in res.items():
        v = 1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']); tot[key] = tot.get(key, 0) + v
        out.append(f"k{key[0]} m{key[1]}: {v:+.4f}")
    print(name, " | ".join(out), flush=True)
print("sum:", " | ".join(f"k{k[0]} m{k[1]} {v:+.4f}" for k, v in sorted(tot.items(), key=lambda x: -x[1])), flush=True)
