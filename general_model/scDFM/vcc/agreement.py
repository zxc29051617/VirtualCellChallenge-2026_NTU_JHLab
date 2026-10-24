"""How well a perturbation's response in one context predicts it in another (pseudobulk, scorer space).

For every pair of contexts and every shared perturbation: Pearson correlation of the common-mode-removed
deltas (delta minus the context's mean delta over its perturbations) over genes measured and expressed in
both contexts (control e > MIN_E), target genes excluded. Reported per pair: number of shared perturbations,
median correlation over all of them and over the 'responsive' half (largest noise-normalised effect energy).

    python -m vcc.agreement [contexts...]
"""
import sys
import numpy as np
from vcc.contexts import load_all

MIN_E = 1.0


def centred(c):
    return c.delta - np.nanmean(c.delta, 0, keepdims=True)


def energy(c):
    z2 = np.where(np.isnan(c.delta), 0, c.delta ** 2 / np.maximum(c.var, 1e-6))
    return z2.sum(1) / np.maximum((~np.isnan(c.delta)).sum(1), 1)          # mean z^2 per gene (1 = noise)


def pair(a, b, targets):
    shared = sorted(set(a.perts) & set(b.perts))
    if len(shared) < 5: return None
    ia = {p: i for i, p in enumerate(a.perts)}; ib = {p: i for i, p in enumerate(b.perts)}
    g = a.measured & b.measured & (a.ctrl > MIN_E) & (b.ctrl > MIN_E) & ~np.isin(a.genes, list(targets))
    A, B = centred(a)[:, g], centred(b)[:, g]
    ea, eb = energy(a), energy(b)
    r = []; w = []
    for p in shared:
        x, y = A[ia[p]], B[ib[p]]; ok = ~(np.isnan(x) | np.isnan(y))
        r.append(np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 50 else np.nan); w.append(min(ea[ia[p]], eb[ib[p]]))
    r, w = np.array(r), np.array(w); top = w >= np.nanmedian(w)
    return len(shared), int(g.sum()), np.nanmedian(r), np.nanmedian(r[top])


if __name__ == "__main__":
    C = load_all(sys.argv[1:] or None)
    targets = set().union(*[set(c.perts) for c in C.values()])
    names = list(C)
    print(f"{'pair':28s} {'shared':>6s} {'genes':>6s} {'r_all':>6s} {'r_resp':>6s}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out = pair(C[a], C[b], targets)
            if out: print(f"{a + ' ~ ' + b:28s} {out[0]:6d} {out[1]:6d} {out[2]:6.3f} {out[3]:6.3f}", flush=True)
