"""Per-(source, target) weights in the trans average, screened on the three hold-outs (2026-10-20).

    python weights_screen.py [H1 KOLF200 HCT116]          (pertrelay venv)

The challenge states every target is knocked down > 80 %; the sources' own knockdown of a target varies a lot (X-Atlas
median ~-0.3 ln, H1 ~-1.9). The model rescales strong knockdowns down (kscale, capped at 1) but weights every source
equally. Variants weight each (source, target) row by
  kd^g     kd = min(|own_obs| / 1.9, 1) (g = 1, 2): weaker knockdown -> weaker, noisier signature
  depth    ncell / (ncell + n0) (n0 = 50, 200)
with or without per-source PC cleaning (k = 10). Totals = own + cis + s x trans after the usual shared-component and
control-axis removal, matched norm; scored by bulk_eval (pds proxy, mse_free). No hold-out data in the weights.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def trans(C, train, targets, kclean, wfun):
    ti = {t: i for i, t in enumerate(targets)}; num = np.zeros((len(targets), S.G)); den = np.zeros((len(targets), S.G))
    for c in train:
        Y = C[c]["Ysrc"].astype(np.float64); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        if kclean > 0:
            _, _, vt = np.linalg.svd(Yz - Yz.mean(0, keepdims=True), full_matrices=False)
            Yz = np.where(ok, Yz - (Yz @ vt[:kclean].T) @ vt[:kclean], 0)
        w = wfun(C[c])
        for j, t in enumerate(C[c]["targets"]):
            i = ti.get(t)
            if i is not None: num[i] += w[j] * Yz[j]; den[i] += w[j] * ok[j]
    return S.ALPHA * num / (den + S.LAM) / np.log(2)


def kd(d):
    o = np.nan_to_num(d["own_obs"].astype(np.float64), nan=0.0); return np.clip(np.abs(np.minimum(o, 0)) / 1.9, 0, 1)


WEIGHTS = {"equal": lambda d: np.ones(len(d["targets"])), "kd^1": lambda d: kd(d), "kd^2": lambda d: kd(d) ** 2,
           "depth50": lambda d: d["ncell"] / (d["ncell"] + 50.0), "depth200": lambda d: d["ncell"] / (d["ncell"] + 200.0),
           "kd^1*depth50": lambda d: kd(d) * d["ncell"] / (d["ncell"] + 50.0)}


def main():
    res = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        if name == (sys.argv[1:] or ["H1"])[0]:
            for c in train: print(f"  {c:13s} median kd {np.median(kd(C[c])):.2f}  frac kd<0.5 {np.mean(kd(C[c]) < 0.5):.2f}", flush=True)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        prep = lambda t: (lambda t5: t5 - (t5 @ Vc.T) @ Vc)(E.remove_shared(t).astype(np.float64))
        base_norm = np.linalg.norm(prep(B["full"] - oc)); row = {}
        for kclean in [0, 10]:
            for wn, wf in WEIGHTS.items():
                t8 = prep(trans(C, train, targets, kclean, wf)); nrm = np.linalg.norm(t8) / base_norm
                for s in [0.1, 0.3]: row[f"clean{kclean} {wn} s{s}"] = BE.score(B, oc + (s / nrm) * t8)
                print(f"  {name} clean{kclean} {wn:13s} " + "  ".join(f"s{s}: pds {row[f'clean{kclean} {wn} s{s}']['pds']:.4f} mse {row[f'clean{kclean} {wn} s{s}']['mse_free']:.4f}" for s in [0.1, 0.3]), flush=True)
        res[name] = row
    json.dump(res, open("weights_screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
