"""Joint grid for the refined totals (2026-10-20, cell-free): gamma (|t|^(gamma-1)), power of the (agreement x n) factor,
cis scale; entry agreement k 1 throughout. Approx value 1.94 d(pds) + 0.95 d(cos) vs the G3c setting (gamma 3, power 0.5,
cis 1), summed over the three hold-outs, and the participation ratio (concentration) per set."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W, energy_proxy_screen as EP, agree_totals_screen as AT
SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"
grid = [(3, 0.5, 1), (3, 0.5, 3), (2, 1.0, 1), (2, 1.0, 3), (3, 0.75, 3), (2, 0.5, 3), (4, 0.5, 3)]
tot = {g: 0.0 for g in grid}; prs = {g: [] for g in grid}
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
    a, n = EP.agreement(C, train, targets); an = (np.maximum(a, 0) + 1e-3) * n / n.mean()
    t5 = E.remove_shared(td.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V; nr = np.maximum(np.linalg.norm(t8, axis=1), 1e-9)
    res = {}
    for (g, pw, cs) in grid:
        t = t8 * (nr ** (g - 1) * an ** pw)[:, None]; t *= 0.6 * bn / np.linalg.norm(t); e = (t ** 2).sum(1)
        res[(g, pw, cs)] = BE.score(B, own + cs * cis + t); prs[(g, pw, cs)].append(e.sum() ** 2 / (e ** 2).sum())
    ref = res[(3, 0.5, 1)]
    for key, r in res.items(): tot[key] += 1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos'])
    print(name, "done", flush=True)
for key in grid: print(f"gamma {key[0]} power {key[1]} cis x{key[2]}: sum {tot[key]:+.4f}  PR " + "/".join(f"{p:.0f}" for p in prs[key]), flush=True)
