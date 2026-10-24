"""Per-entry inverse-variance weights in the source combination (2026-10-20, cell-free).

    python ivw_screen.py [H1 KOLF200 HCT116]          (scDFM venv)

The G1 depth weight n/(n + 200) is per (source, target) row. Here each entry gets w = 1 / (s2 + tau2) with the Poisson
sampling variance of the source's log ratio s2 ~ 1 / (ncell x m_g) + 1 / (n_ctrl x m_g), m_g = expected counts per cell of
gene g in that source's controls (CP50k x 0.2, i.e. ~10k UMI per cell; n_ctrl taken as 2,000) and tau2 the between-context
heterogeneity (0.02, 0.05, 0.2). Then the G1 processing (shared + control axes removed, gamma 3). cos / pds / approx value.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"


def trans(C, train, targets, tau2):
    ti = {t: i for i, t in enumerate(targets)}; num = np.zeros((len(targets), S.G)); den = np.zeros((len(targets), S.G))
    for c in train:
        d = C[c]; Y = d["Ysrc"].astype(np.float64); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        m = np.maximum(d["ctrl"].astype(np.float64) * 0.2, 0.01)
        for j, t in enumerate(d["targets"]):
            i = ti.get(t)
            if i is None: continue
            if tau2 is None: w = d["ncell"][j] / (d["ncell"][j] + 200.0)
            else:
                s2 = 1.0 / (d["ncell"][j] * m) + 1.0 / (2000.0 * m); w = (1.0 / (s2 + tau2)) * tau2      # normalised: -> 1 when noise-free
            num[i] += w * Yz[j]; den[i] += w * ok[j]
    return S.ALPHA * num / (den + S.LAM) / np.log(2)


def main():
    tot = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        def sc(t):
            t5 = E.remove_shared(t.astype(np.float32)).astype(np.float64); t8 = t5 - (t5 @ V.T) @ V
            t8 = t8 * (np.maximum(np.linalg.norm(t8, axis=1), 1e-9) ** 2)[:, None]
            return BE.score(B, oc + 0.6 * t8 * (bn / np.linalg.norm(t8)))
        ref = sc(trans(C, train, targets, None)); out = [f"G1 cos {ref['cos']:+.4f} pds {ref['pds']:.4f}"]
        for tau2 in [0.02, 0.05, 0.2]:
            r = sc(trans(C, train, targets, tau2)); v = 1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']); tot[tau2] = tot.get(tau2, 0) + v
            out.append(f"tau2 {tau2}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {v:+.4f}")
        print(name, " | ".join(out), flush=True)
    print("sum:", " | ".join(f"tau2 {k}: {v:+.4f}" for k, v in tot.items()), flush=True)


if __name__ == "__main__":
    main()
