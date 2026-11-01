"""Per-entry sign agreement in the TOTALS (2026-10-20, cell-free): trans_pg x (agreement_pg / median)^k before the G1
processing (shared + control axes removed, gamma 3 sharpening), k = 0 (G1), 0.5, 1, 2; with and without the per-target
(a n)^0.5 factor. Reported: global noise-corrected cosine, pds, approx value vs G1. Training sources only."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W, energy_proxy_screen as EP

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def entry_agreement(C, train, targets, shape):
    ti = {t: i for i, t in enumerate(targets)}; n = np.zeros(shape); s1 = np.zeros(shape); pos = np.zeros(shape)
    for c in train:
        Y = C[c]["Ysrc"]
        for j, t in enumerate(C[c]["targets"]):
            i = ti.get(t)
            if i is None: continue
            y = Y[j]; ok = np.isfinite(y); yz = np.where(ok, y, 0); n[i] += ok; s1[i] += yz; pos[i] += yz > 0
    mean = s1 / np.maximum(n, 1)
    return np.where(mean > 0, pos / np.maximum(n, 1), 1 - pos / np.maximum(n, 1))


def main():
    tot = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        td = W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]); ag = entry_agreement(C, train, targets, td.shape)
        med = np.median(ag[np.abs(td) > 0]); a, n = EP.agreement(C, train, targets); f = np.sqrt((np.maximum(a, 0) + 1e-3) * n / n.mean())
        def sc(t, fac=None):
            t5 = E.remove_shared(t.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
            t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2)[:, None]
            if fac is not None: t8 = t8 * fac[:, None]
            return BE.score(B, oc + 0.6 * t8 * (bn / np.linalg.norm(t8)))
        ref = sc(td); out = [f"G1 cos {ref['cos']:+.4f} pds {ref['pds']:.4f}"]
        for lab, k, fac in [("k0.5", 0.5, None), ("k1", 1.0, None), ("k2", 2.0, None), ("M4h", 0.0, f), ("M4h+k1", 1.0, f)]:
            r = sc(td * np.clip(ag / med, 0, None) ** k, fac)
            v = 1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']); tot[lab] = tot.get(lab, 0) + v
            out.append(f"{lab}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {v:+.4f}")
        print(name, " | ".join(out), flush=True)
    print("sum:", " | ".join(f"{k} {v:+.4f}" for k, v in tot.items()), flush=True)


if __name__ == "__main__":
    main()
