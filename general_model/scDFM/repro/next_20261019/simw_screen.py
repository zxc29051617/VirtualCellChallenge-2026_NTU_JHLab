"""Context-similarity source weights re-checked under the G1 objective (2026-10-20, cell-free): weight source c by
exp(tau (r_c - 1)), r_c = Pearson correlation of log control profiles (CP50k) between source c and the new context over
genes expressed in either (computed from controls only), on top of the G1 depth weights. cos / pds / approx value."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"


def trans(C, train, targets, wc):
    ti = {t: i for i, t in enumerate(targets)}; num = np.zeros((len(targets), S.G)); den = np.zeros((len(targets), S.G))
    for c in train:
        d = C[c]; Y = d["Ysrc"].astype(np.float64); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        wr = wc[c] * d["ncell"] / (d["ncell"] + 200.0)
        for j, t in enumerate(d["targets"]):
            i = ti.get(t)
            if i is not None: num[i] += wr[j] * Yz[j]; den[i] += wr[j] * ok[j]
    return S.ALPHA * num / (den + S.LAM) / np.log(2)


for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
    train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
    B = BE.load(name); oc = B["oc"]; targets = B["targets"]; x_new = np.log1p(B["rate"] * 5e4)
    r = {}
    for c in train:
        xc = np.log1p(C[c]["ctrl"].astype(np.float64)); m = C[c]["measured"] & ((xc > 0.4) | (x_new > 0.4))
        r[c] = float(np.corrcoef(xc[m], x_new[m])[0, 1])
    V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
    bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
    def sc(t):
        t5 = E.remove_shared(t.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
        t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2)[:, None]
        return BE.score(B, oc + 0.6 * t8 * (bn / np.linalg.norm(t8)))
    ref = sc(trans(C, train, targets, {c: 1.0 for c in train}))
    out = ["r: " + ", ".join(f"{c} {r[c]:.2f}" for c in sorted(r, key=lambda c: -r[c])[:4]), f"G1 cos {ref['cos']:+.4f} pds {ref['pds']:.4f}"]
    for tau in [2.0, 5.0, 10.0]:
        res = sc(trans(C, train, targets, {c: float(np.exp(tau * (r[c] - 1))) for c in train}))
        out.append(f"tau{tau}: cos {res['cos']:+.4f} pds {res['pds']:.4f} val {1.94 * (res['pds'] - ref['pds']) + 0.95 * (res['cos'] - ref['cos']):+.4f}")
    print(name, " | ".join(out), flush=True)
