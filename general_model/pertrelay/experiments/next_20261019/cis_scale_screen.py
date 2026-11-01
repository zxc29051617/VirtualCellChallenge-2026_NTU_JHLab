"""Cis strength in the refined totals (2026-10-20, cell-free): totals = own + g x cis + trans (refined: entry agreement,
(a n)^0.5 x |t|^2, matched norm), g = 0.5 / 1 / 1.5 / 2. Approx value vs g 1."""
import json, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W, energy_proxy_screen as EP, agree_totals_screen as AT
from vcc.gm import Genome
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
gen = Genome(S.GENES); tot = {}
for name in ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]
    own = np.zeros_like(oc)
    for i, t in enumerate(targets):
        k = S.GI.get(t)
        if k is not None: own[i, k] = oc[i, k]
    cis = oc - own
    V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
    td = W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]); ag = AT.entry_agreement(C, train, targets, td.shape)
    td = td * np.clip(ag / np.median(ag[np.abs(td) > 0]), 0, None)
    a, n = EP.agreement(C, train, targets); f = np.sqrt((np.maximum(a, 0) + 1e-3) * n / n.mean())
    t5 = E.remove_shared(td.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
    t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2 * f)[:, None]; t8 *= 0.6 * bn / np.linalg.norm(t8)
    res = {g: BE.score(B, own + g * cis + t8) for g in [1.0, 2.0, 3.0, 4.0]}
    out = []
    for g, r in res.items():
        v = 1.94 * (r['pds'] - res[1.0]['pds']) + 0.95 * (r['cos'] - res[1.0]['cos']); tot[g] = tot.get(g, 0) + v
        out.append(f"g{g}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {v:+.4f}")
    print(name, " | ".join(out), flush=True)
print("sum:", " | ".join(f"g{k} {v:+.4f}" for k, v in tot.items()), flush=True)
