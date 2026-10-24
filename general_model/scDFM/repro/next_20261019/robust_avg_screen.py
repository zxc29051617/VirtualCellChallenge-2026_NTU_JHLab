"""Robust combination of the sources' responses per target (2026-10-20, cell-free): does down-weighting a source that
disagrees with the others (for that target) improve the direction? Variants (all with the G1 depth weights and the G1
processing: shared + control axes removed, gamma 3, matched norm):
  loo w^k   each (source, target) row weighted by max(0, cos(row, mean of the other sources' rows))^k  (k = 1, 2)
  median    per-gene weighted median across the sources (no ridge), x the ridge's shrink factor n/(n + 3)
  trim      per-gene mean after dropping the largest and smallest source value when >= 5 sources
Reported: global noise-corrected cosine, pds, approx value 1.94 d(pds) + 0.95 d(cos) vs G1. Training sources only.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import bulk_eval as BE, render_eval as E, sigexp as S

SIGDIR = "/home/zxc29051617/VCC/scDFM/repro/signature_20261013"


def stacks(C, train, targets):
    ti = {t: i for i, t in enumerate(targets)}; rows = {i: [] for i in range(len(targets))}
    for c in train:
        d = C[c]; wr = d["ncell"] / (d["ncell"] + 200.0)
        for j, t in enumerate(d["targets"]):
            i = ti.get(t)
            if i is not None: rows[i].append((wr[j], d["Ysrc"][j].astype(np.float64)))
    return rows


def combine(rows, n_t, mode, k=1):
    out = np.zeros((n_t, S.G))
    for i, lst in rows.items():
        if not lst: continue
        W = np.array([w for w, _ in lst]); Y = np.stack([y for _, y in lst]); ok = np.isfinite(Y); Yz = np.where(ok, Y, 0)
        if mode == "loo" and len(lst) >= 3:
            tot = (W[:, None] * Yz).sum(0)
            for s in range(len(lst)):
                rest = tot - W[s] * Yz[s]; m = ok[s]
                c = float(Yz[s][m] @ rest[m] / (np.linalg.norm(Yz[s][m]) * np.linalg.norm(rest[m]) + 1e-12))
                W[s] *= max(c, 0.0) ** k + 1e-3
        if mode in ("ridge", "loo"):
            out[i] = (W[:, None] * Yz).sum(0) / ((W[:, None] * ok).sum(0) + S.LAM)
        elif mode == "median":
            Ym = np.where(ok, Y, np.nan); med = np.nan_to_num(np.nanmedian(Ym, axis=0)); n = ok.sum(0)
            out[i] = med * n / (n + S.LAM)
        elif mode == "trim":
            Ym = np.where(ok, Y, np.nan); n = ok.sum(0)
            if len(lst) >= 5:
                srt = np.sort(np.where(ok, Y, np.nan), axis=0)          # NaN last
                lo = np.zeros(S.G, int); hi = n - 1
                mask = np.ones_like(srt, bool); idx = np.arange(srt.shape[0])[:, None]
                mask &= (idx > lo[None]) & (idx < hi[None]) | (n[None] < 5)
                val = np.where(mask & np.isfinite(srt), srt, 0); cnt = (mask & np.isfinite(srt)).sum(0)
                out[i] = val.sum(0) / (cnt + S.LAM)
            else:
                out[i] = np.nansum(Ym, 0) / (n + S.LAM)
    return S.ALPHA * out / np.log(2)


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
        rows = stacks(C, train, targets)
        ref = sc(combine(rows, len(targets), "ridge")); out = [f"G1 cos {ref['cos']:+.4f} pds {ref['pds']:.4f}"]
        for lab, mode, k in [("loo w^1", "loo", 1), ("loo w^2", "loo", 2), ("median", "median", 1), ("trim", "trim", 1)]:
            r = sc(combine(rows, len(targets), mode, k))
            out.append(f"{lab}: cos {r['cos']:+.4f} pds {r['pds']:.4f} val {1.94 * (r['pds'] - ref['pds']) + 0.95 * (r['cos'] - ref['cos']):+.4f}")
        print(name, " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
