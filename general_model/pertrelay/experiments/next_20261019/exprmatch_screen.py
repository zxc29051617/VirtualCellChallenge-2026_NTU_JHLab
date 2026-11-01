"""Per-(source, gene) weights by expression match with the NEW context (2026-10-20, cell-free).

    python exprmatch_screen.py [H1 KOLF200 HCT116]          (pertrelay venv)

A source's response of gene g is more trustworthy for the new context when g is expressed at a similar level in both
(the new context's level comes from its own controls). Weight w_cg = exp(-|ln((x_c[g] + e) / (x_new[g] + e))| / tau),
x = control CP50k, e = 0.5, on top of the G1 depth weights; then the G1 processing (shared + control axes removed, gamma 3).
Reported: global noise-corrected cosine, pds, approx value vs G1. Hold-out truth for scoring only.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def trans(C, train, targets, x_new, tau):
    ti = {t: i for i, t in enumerate(targets)}; num = np.zeros((len(targets), S.G)); den = np.zeros((len(targets), S.G))
    for c in train:
        d = C[c]; Y = d["Ysrc"].astype(np.float64); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        wg = np.ones(S.G) if tau is None else np.exp(-np.abs(np.log((d["ctrl"].astype(np.float64) + 0.5) / (x_new + 0.5))) / tau)
        wr = d["ncell"] / (d["ncell"] + 200.0)
        for j, t in enumerate(d["targets"]):
            i = ti.get(t)
            if i is not None: num[i] += wr[j] * wg * Yz[j]; den[i] += wr[j] * wg * ok[j]
    return S.ALPHA * num / (den + S.LAM) / np.log(2)


def main():
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        x_new = B["rate"] * 5e4                                           # CP50k in the hold-out's controls
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        def sc(t):
            t5 = E.remove_shared(t.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
            t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2)[:, None]
            return BE.score(B, oc + 0.6 * t8 * (bn / np.linalg.norm(t8)))
        ref = sc(trans(C, train, targets, x_new, None)); out = [f"G1 cos {ref['cos']:+.4f} pds {ref['pds']:.4f}"]
        for tau in [0.5, 1.0, 2.0, 4.0]:
            r = sc(trans(C, train, targets, x_new, tau))
            out.append(f"tau{tau}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']):+.4f}")
        print(name, " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
