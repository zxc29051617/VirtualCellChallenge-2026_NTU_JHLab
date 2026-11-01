"""Combined per-target magnitude rules for the totals (2026-10-20, cell-free). Base: G1 (|t|^3 after the removals).
Extra factors, each from training sources or external screening data only: agreement a (mean pairwise source cosine),
n (number of sources, / mean), DepMap fraction of lines where the target is essential (frac). Reported: cos / pds / approx
value 1.94 d(pds) + 0.95 d(cos) vs G1 on the three hold-outs.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W, energy_proxy_screen as EP, depmap_energy as DE

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def main():
    allt = set()
    for name in ["H1", "KOLF200", "HCT116"]: allt |= set(json.load(open(f"/home/zxc29051617/VCC/data/stage3/loco/{name}/real.json"))["targets"])
    dm = DE.depmap_effects(sorted(allt))
    tot = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        frac = np.array([dm.get(t, (np.nan, np.nan))[1] for t in targets]); frac = np.where(np.isfinite(frac), frac, np.nanmedian(frac))
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        t5 = E.remove_shared(W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]).astype(np.float32)).astype(np.float64)
        t8 = t5 - (t5 @ V.T) @ V; nr = np.maximum(np.linalg.norm(t8, axis=1), 1e-9); unit = t8 / nr[:, None]
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        a, n = EP.agreement(C, train, targets); ap = np.maximum(a, 0) + 1e-3; nn = n / n.mean()
        rules = {"G1": nr ** 3, "M4 a+n": nr ** 3 * ap * nn, "M4h (a n)^0.5": nr ** 3 * np.sqrt(ap * nn),
                 "M4g2 |t|^2 a n": nr ** 2 * ap * nn, "M4q (a n)^0.25": nr ** 3 * (ap * nn) ** 0.25}
        out = []; ref = None
        for k, e in rules.items():
            t = unit * e[:, None]; r = BE.score(B, oc + 0.6 * t * (bn / np.linalg.norm(t)))
            if ref is None: ref = r
            v = 1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']); tot[k] = tot.get(k, 0) + v
            pr = ((e ** 2).sum() ** 2) / ((e ** 4).sum())
            out.append(f"{k}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {v:+.4f} PR {pr:.0f}")
        print(name, " | ".join(out), flush=True)
    print("sum over sets:", " | ".join(f"{k} {v:+.4f}" for k, v in tot.items()), flush=True)


if __name__ == "__main__":
    main()
