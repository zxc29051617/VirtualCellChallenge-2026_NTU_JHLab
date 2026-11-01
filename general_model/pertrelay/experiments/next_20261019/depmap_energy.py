"""Does essentiality (DepMap 24Q4 CRISPR gene effect) tell which targets respond strongly in a new line? (2026-10-20)

    python depmap_energy.py          (pertrelay venv)

For each hold-out: Spearman of the true noise-corrected energy per target with -mean gene effect across all DepMap lines
(pan-essentiality) and with the fraction of lines where the gene is essential (effect < -0.5); then the global cosine /
pds when the G1 per-target magnitude |t|^3 is multiplied by (1 + b x essentiality) for a few b. No hold-out data in the
proxies (DepMap is cell-line screening data, not the hold-out's perturbation outcomes).
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S, weights_screen as W

SIGDIR = "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013"
DM = "/home/zxc29051617/VCC/data/external/depmap/CRISPRGeneEffect.csv"


def depmap_effects(genes):
    head = pd.read_csv(DM, nrows=0).columns
    sym = {c.split(" (")[0]: c for c in head[1:]}
    cols = [sym[g] for g in genes if g in sym]
    X = pd.read_csv(DM, usecols=[head[0]] + cols)
    eff = {c.split(" (")[0]: X[c].to_numpy(dtype=np.float64) for c in cols}
    return {g: (-np.nanmean(v), np.nanmean(v < -0.5)) for g, v in eff.items()}


def main():
    allt = set()
    for name in ["H1", "KOLF200", "HCT116"]: allt |= set(json.load(open(f"/home/zxc29051617/VCC/data/stage3/loco/{name}/real.json"))["targets"])
    dm = depmap_effects(sorted(allt)); print(f"DepMap effects for {len(dm)}/{len(allt)} hold-out targets", flush=True)
    for name in ["H1", "KOLF200", "HCT116"]:
        train = json.load(open(f"{SIGDIR}/weights_{name}.json"))["train"]; C = S.load_cache(train)
        B = BE.load(name); oc = B["oc"]; targets = B["targets"]; D = B["D"].copy()
        for i, k in enumerate(B["own"]):
            if k >= 0: D[i, k] = 0
        energy = (D ** 2).sum(1) - B["v"]
        ess = np.array([dm.get(t, (np.nan, np.nan))[0] for t in targets]); frac = np.array([dm.get(t, (np.nan, np.nan))[1] for t in targets])
        ok = np.isfinite(ess); ess = np.where(ok, ess, np.nanmedian(ess)); frac = np.where(ok, frac, np.nanmedian(frac))
        V = E.control_axes(B["C"], B["lib"], 50).astype(np.float64)
        t5 = E.remove_shared(W.trans(C, train, targets, 0, W.WEIGHTS["depth200"]).astype(np.float32)).astype(np.float64)
        t8 = t5 - (t5 @ V.T) @ V; nr = np.maximum(np.linalg.norm(t8, axis=1), 1e-9); unit = t8 / nr[:, None]
        bn = np.linalg.norm(E.remove_shared((B["full"] - oc).astype(np.float32)))
        out = [f"spearman(energy, -effect) {spearmanr(ess[ok], energy[ok]).correlation:.3f}, (energy, frac ess) {spearmanr(frac[ok], energy[ok]).correlation:.3f}, (energy, |t|) {spearmanr(nr, energy).correlation:.3f}, (energy, |t| x (1+2 frac)) {spearmanr(nr * (1 + 2 * frac), energy).correlation:.3f}"]
        ref = None
        for b in [0.0, 1.0, 3.0, 10.0]:
            e = nr ** 3 * (1 + b * frac); t = unit * e[:, None]; r = BE.score(B, oc + 0.6 * t * (bn / np.linalg.norm(t)))
            if ref is None: ref = r
            out.append(f"b{b}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']):+.4f}")
        print(name, " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
