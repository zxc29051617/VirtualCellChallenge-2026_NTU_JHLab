"""Per-source cleaning before averaging, screened on the three hold-outs (2026-10-20).

    python pcclean_screen.py [H1 KOLF200 HCT116]          (pertrelay venv)

Each training source's response matrix (cached Ysrc: its targets x genes, ln ratios with its own mean response
removed) has dominant programs of its own (cell cycle, stress, the screen's generic response). Variant k removes the
top-k principal components of EACH source's matrix (computed over all its cached targets) from its rows, then the
usual ridge average (lam 3, alpha 3), shared-component and control-axis removal; totals = own + cis + s x trans.
k = 0 reproduces the current model. Scored by bulk_eval (pds proxy, mse_free). No hold-out data in the cleaning.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"


def cleaned_trans(C, train, targets, k):
    num = np.zeros((len(targets), S.G)); den = np.zeros((len(targets), S.G)); ti = {t: i for i, t in enumerate(targets)}
    for c in train:
        Y = C[c]["Ysrc"].astype(np.float64); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        if k > 0:
            _, _, vt = np.linalg.svd(Yz - Yz.mean(0, keepdims=True), full_matrices=False)
            Yz = np.where(ok, Yz - (Yz @ vt[:k].T) @ vt[:k], 0)
        for j, t in enumerate(C[c]["targets"]):
            i = ti.get(t)
            if i is not None: num[i] += Yz[j]; den[i] += ok[j]
    return S.ALPHA * num / (den + S.LAM) / np.log(2)                 # log2, like components()


def main():
    res = {}
    for name in sys.argv[1:] or ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        prep = lambda t: (lambda t5: t5 - (t5 @ V.T) @ V)(E.remove_shared(t).astype(np.float64))
        base_norm = np.linalg.norm(prep(B["full"] - oc)); row = {}
        chk = cleaned_trans(C, train, targets, 0); print(name, "k0 reproduces model trans:", float(np.abs(chk - (B["full"] - oc)).max()), flush=True)
        for k in [0, 3, 10, 20, 40]:
            t8 = prep(cleaned_trans(C, train, targets, k)); nrm = np.linalg.norm(t8) / base_norm
            for s in [0.1, 0.3, 0.5]:
                row[f"k{k} s{s}"] = BE.score(B, oc + (s / nrm) * t8)
            print(f"  k{k:<3d} " + "  ".join(f"s{s}: pds {row[f'k{k} s{s}']['pds']:.4f} mse {row[f'k{k} s{s}']['mse_free']:.4f}" for s in [0.1, 0.3, 0.5]), flush=True)
        res[name] = row
    json.dump(res, open("pcclean_screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
