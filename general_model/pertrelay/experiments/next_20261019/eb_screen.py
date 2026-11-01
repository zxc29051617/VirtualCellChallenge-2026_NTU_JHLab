"""Transfer-coefficient (empirical Bayes) shrinkage of the trans signature, screened on the three hold-outs (2026-10-20).

    python eb_screen.py [H1 KOLF200 HCT116]          (pertrelay venv)

Idea: the prediction for a new line should be E[true response | sources]. Within each hold-out's TRAINING contexts,
predict every context from the others (leave its cell-type group out, same ridge as the model) and regress the left-out
context's values on that prediction:
  per gene    w_g = sum_{c,p} s_-c[p,g] y_c[p,g] / (sum s_-c^2 + kappa)       (how much gene g's response transfers)
  per target  a_p = sum_{c,g} s_-c[p,g] y_c[p,g] / sum s_-c^2                 (how well target p's response transfers)
The hold-out prediction's trans part is multiplied by w_g (and/or a_p, clipped to [0, 1.5]) before the usual
shared-component and control-axis removal; totals = own + cis + s x trans, scored by bulk_eval (pds proxy and
mse_free) over a grid of s. No hold-out data enters the weights (they come from the training contexts only).
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

LAM, KAPPA = 3.0, None
SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def transfer_coefs(C, train):
    groups = {}
    for c in train: groups.setdefault(S.GROUP_OF[c], []).append(c)
    allt = sorted({t for c in train for t in C[c]["targets"]})
    ti = {t: i for i, t in enumerate(allt)}
    NUM = np.zeros((len(allt), S.G)); DEN = np.zeros((len(allt), S.G))
    GN = {g: np.zeros_like(NUM) for g in groups}; GD = {g: np.zeros_like(DEN) for g in groups}
    for c in train:
        g = S.GROUP_OF[c]; Y = C[c]["Ysrc"]; ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        idx = [ti[t] for t in C[c]["targets"]]
        NUM[idx] += Yz; DEN[idx] += ok; GN[g][idx] += Yz; GD[g][idx] += ok
    gnum = np.zeros(S.G); gden = np.zeros(S.G); tnum = np.zeros(len(allt)); tden = np.zeros(len(allt))
    for c in train:
        g = S.GROUP_OF[c]; Y = C[c]["Ysrc"]; idx = [ti[t] for t in C[c]["targets"]]
        X = (NUM[idx] - GN[g][idx]) / (DEN[idx] - GD[g][idx] + LAM)
        ok = np.isfinite(Y) & ((DEN[idx] - GD[g][idx]) > 0); Yz = np.where(ok, Y, 0); Xz = np.where(ok, X, 0)
        gnum += (Xz * Yz).sum(0); gden += (Xz ** 2).sum(0)
        np.add.at(tnum, idx, (Xz * Yz).sum(1)); np.add.at(tden, idx, (Xz ** 2).sum(1))
    return allt, gnum, gden, tnum, tden


def main():
    res = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]
        C = S.load_cache(train); allt, gnum, gden, tnum, tden = transfer_coefs(C, train)
        w0 = gnum.sum() / gden.sum()
        B = BE.load(name); oc = B["oc"]; tr = B["full"] - oc; targets = B["targets"]
        kappa = np.median(gden[gden > 0])                       # shrink each gene's coefficient toward the global one
        wg = np.clip((gnum + kappa * w0) / (gden + kappa), 0, 1.5)
        ap = np.array([np.clip(tnum[allt.index(t)] / max(tden[allt.index(t)], 1e-9), 0, 1.5) if t in allt else w0 for t in targets])
        print(name, f"global coef {w0:.3f}; gene coef median {np.median(wg):.3f} IQR {np.percentile(wg, 25):.3f}-{np.percentile(wg, 75):.3f}; "
              f"target coef median {np.median(ap):.3f} IQR {np.percentile(ap, 25):.3f}-{np.percentile(ap, 75):.3f}", flush=True)
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        prep = lambda t: (lambda t5: t5 - (t5 @ V.T) @ V)(E.remove_shared(t).astype(np.float64))
        variants = {"base": tr, "gene": tr * wg[None], "target": tr * ap[:, None], "gene+target": tr * wg[None] * ap[:, None]}
        row = {}
        for vn, t in variants.items():
            t8 = prep(t); nrm = np.linalg.norm(t8) / np.linalg.norm(prep(tr))        # s is relative to base norm
            for s in [0.05, 0.1, 0.2, 0.3, 0.5]:
                r = BE.score(B, oc + (s / nrm) * t8); row[f"{vn} s{s}"] = r
            best = min((v["mse_free"], k) for k, v in row.items() if k.startswith(vn + " "))
            print(f"  {vn:12s} " + "  ".join(f"s{k.split(' s')[1]}: pds {v['pds']:.4f} mse {v['mse_free']:.4f}" for k, v in row.items() if k.startswith(vn + " ")), flush=True)
        res[name] = {"global_coef": w0, "rows": row}
    json.dump(res, open("eb_screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
