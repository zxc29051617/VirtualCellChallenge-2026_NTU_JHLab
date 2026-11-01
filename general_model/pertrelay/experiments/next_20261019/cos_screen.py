"""Global noise-corrected cosine (what the mse member rewards at the budget = claim point) and pds for trans-part
variants on the three hold-outs (2026-10-20, cell-free).

    python cos_screen.py [H1 KOLF200 HCT116]          (pertrelay venv)

Variants of the trans part (all followed by shared-component and control-axis removal; totals = own + cis + 0.6 x trans
at matched norm):
  base / clean10 / depth200 / clean10+depth200     source combination (see pcclean_screen.py, weights_screen.py)
  gamma g     per-target magnitude sharpening: row p scaled by |t_p|^(g - 1) (g = 1 is base; g = 0 all rows equal norm)
  srcmag      row norm set to the mean norm of that target's source responses (how strongly it responds elsewhere)
No hold-out data in any variant.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def srcmag(C, train, targets):
    acc = {t: [] for t in targets}
    for c in train:
        Y = C[c]["Ysrc"]
        for j, t in enumerate(C[c]["targets"]):
            if t in acc:
                y = Y[j]; ok = np.isfinite(y)
                if ok.sum() > 1000: acc[t].append(np.sqrt(np.mean(y[ok] ** 2)))
    return np.array([np.mean(acc[t]) if acc[t] else np.nan for t in targets])


def main():
    res = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        Vc = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        prep = lambda t: (lambda t5: t5 - (t5 @ Vc.T) @ Vc)(E.remove_shared(t).astype(np.float64))
        t8b = prep(B["full"] - oc); base_norm = np.linalg.norm(t8b)
        var = {"base": t8b, "clean10": prep(W.trans(C, train, targets, 10, W.WEIGHTS["equal"])),
               "depth200": prep(W.trans(C, train, targets, 0, W.WEIGHTS["depth200"])),
               "clean10+depth200": prep(W.trans(C, train, targets, 10, W.WEIGHTS["depth200"]))}
        nr = np.linalg.norm(t8b, axis=1)
        for g in [0.0, 0.5, 1.5, 2.0]:
            var[f"gamma{g}"] = t8b * (np.maximum(nr, 1e-9) ** (g - 1))[:, None]
        m = srcmag(C, train, targets); m = np.where(np.isfinite(m), m, np.nanmedian(m))
        var["srcmag"] = t8b / np.maximum(nr, 1e-9)[:, None] * m[:, None]
        row = {}
        for k, t in var.items():
            t = t * (base_norm / np.linalg.norm(t))
            row[k] = BE.score(B, oc + 0.6 * t)
            print(f"  {name} {k:18s} cos {row[k]['cos']:+.4f}  pds {row[k]['pds']:.4f}  mse_free {row[k]['mse_free']:.4f}", flush=True)
        res[name] = row
    json.dump(res, open("cos_screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
