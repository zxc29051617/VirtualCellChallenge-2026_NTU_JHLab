"""Source-combination settings re-checked for the G1 totals objective (2026-10-20, cell-free): global noise-corrected
cosine (mse at budget = claim) and pds on the three hold-outs, trans = depth-weighted ridge -> shared + control axes
removed -> per-target sharpening gamma 3, matched norm. Variants (one change at a time from the G1 setting):
  lam l      ridge strength (G1: 3)
  kcap k     knockdown rescaling cap (G1: 1; k = 2 lets weak-knockdown sources count up to 2x)
  qual q     extra weight quality^q per source (the cache's per-context quality score)
Approximate leaderboard value: 1.94 x d(pds raw) + 0.95 x d(cos) relative to the G1 setting.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def trans(C, train, targets, lam=3.0, kcap=1.0, q=0.0, n0=200.0):
    ti = {t: i for i, t in enumerate(targets)}; num = np.zeros((len(targets), S.G)); den = np.zeros((len(targets), S.G))
    for c in train:
        d = C[c]; Y = d["Ysrc"].astype(np.float64); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        if kcap != 1.0:
            own = d["own_obs"].astype(np.float64); ks_old = d["kscale"].astype(np.float64)
            ks_new = np.where(np.isfinite(own) & (own < -0.3), np.clip(S.OWN / np.where(own < -0.3, own, -1), 0.5, kcap), 1.0)
            Yz = Yz * (ks_new / np.where(ks_old > 0, ks_old, 1.0))[:, None]
        w = d["ncell"] / (d["ncell"] + n0) * (float(d["quality"]) ** q if q else 1.0)
        for j, t in enumerate(d["targets"]):
            i = ti.get(t)
            if i is not None: num[i] += w[j] * Yz[j]; den[i] += w[j] * ok[j]
    return S.ALPHA * num / (den + lam) / np.log(2)


def main():
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        def sc(t):
            t5 = E.remove_shared(t.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
            t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2)[:, None]
            return BE.score(B, oc + 0.6 * t8 * (bn / np.linalg.norm(t8)))
        ref = sc(trans(C, train, targets)); out = [f"G1 cos {ref['cos']:+.4f} pds {ref['pds']:.4f}"]
        for lab, kw in [("lam1", dict(lam=1.0)), ("lam10", dict(lam=10.0)), ("kcap2", dict(kcap=2.0)), ("qual0.5", dict(q=0.5)), ("qual1", dict(q=1.0))]:
            r = sc(trans(C, train, targets, **kw))
            out.append(f"{lab}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']):+.4f}")
        print(name, " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
