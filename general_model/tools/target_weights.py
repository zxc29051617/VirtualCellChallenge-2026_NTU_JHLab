"""Per-target magnitude factors for the totals (2026-10-20): agreement a_p = mean pairwise cosine between the training
sources' responses to p (signature cache: kscale-normalised, common-mode-removed log ratios, own + cis masked) times the
number of sources n_p (/ mean). Uses only the public training screens. Output: npz with targets, w.
    python target_weights.py <targets: pert_counts.csv or effects .npz> <out.npz> [power, default 0.5]
w = (a_p x n_p / mean n)^power; power 0.5 keeps the totals' concentration (participation ratio ~20-30 of 300) in the range
validated on the hold-outs (power 1 gives ~7 on A/B/C).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/zxc29051617/VCC/pertrelay/experiments/signature_20261013")
import sigexp as S
src, out = sys.argv[1:3]; POW = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
targets = [str(t) for t in np.load(src, allow_pickle=True)["targets"]] if src.endswith(".npz") else \
    list(dict.fromkeys(pd.read_csv(src)["target_gene"].astype(str)))
C = S.load_cache([c for c in S.CONTEXTS if c not in set(os.environ.get("VCC_EXCLUDE", "").split(","))])  # VCC_EXCLUDE: mock exams
rows = {t: [] for t in targets}
for c in C:
    Y = C[c]["Ysrc"]
    for j, t in enumerate(C[c]["targets"]):
        if t in rows: rows[t].append(Y[j])
a = np.zeros(len(targets)); n = np.zeros(len(targets))
for i, t in enumerate(targets):
    ys = rows[t]; n[i] = len(ys); cs = []
    for u in range(len(ys)):
        for v in range(u + 1, len(ys)):
            m = np.isfinite(ys[u]) & np.isfinite(ys[v])
            if m.sum() > 500:
                x, y = ys[u][m], ys[v][m]; cs.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    a[i] = np.mean(cs) if cs else 0.0
w = ((np.maximum(a, 0) + 1e-3) * n / max(n[n > 0].mean() if (n > 0).any() else 1.0, 1e-9)) ** POW
miss = n == 0                                      # targets absent from the signature cache (e.g. a changed final panel)
if miss.any():
    w[miss] = np.median(w[~miss]) if (~miss).any() else 1.0
    print(f"WARNING: {int(miss.sum())} targets not in the signature cache -> median weight", flush=True)
np.savez_compressed(out, targets=np.array(targets), w=w.astype(np.float64), agreement=a, n_sources=n)
print(f"wrote {out}: {len(targets)} targets; agreement median {np.median(a):.3f}, n_sources median {np.median(n):.0f}, w range {w.min():.4f}-{w.max():.3f}")
